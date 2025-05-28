import cv2
import time
import os
import logging # Added
# import argparse # No longer directly needed for args namespace
from ultralytics import YOLO
from deepface import DeepFace
from ..config import AppSettings # Import AppSettings
from .utils import draw_detections, draw_info_overlay, iou_calc_for_association
import torch
import numpy as np
from .sort_tracker import Sort 

BYTETRACK_AVAILABLE = False
BYTETracker = None

# Initialize logger for this module
logger = logging.getLogger(__name__) # Added

try:
    from .byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
    logger.info("Successfully imported BYTETracker from local './detector/byte_tracker.py'.") # Logging
except ImportError as e:
    BYTETRACK_AVAILABLE = False
    BYTETracker = None
    logger.warning(f"Failed to import BYTETracker from './detector/byte_tracker.py': {e}. Ensure ByteTrack files (byte_tracker.py, basetrack.py, kalman_filter.py, matching.py) are present in the 'detector/' directory and all dependencies (like cython_bbox) are installed.", exc_info=True) # Logging
    logger.warning("ByteTrack will be unavailable. Falling back to SORT if ByteTrack is selected as tracker type.") # Logging

import threading
import queue
import argparse # Keep for ByteTrack internal arg parsing if needed

class ObjectDetector:
    def __init__(self, config: AppSettings):
        self.config = config
        self.yolo_model = self._load_yolo_model()
        self.cap = self._init_video_capture()
        self.opencv_window_name = "Multi-Analysis Detector (OpenCV)"
        
        if self.cap:
            self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap_fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
            self.effective_fps_for_tracker = self.cap_fps_prop if self.cap_fps_prop > 0 else 30.0
            logger.info(f"Video source opened: {self.cap_w}x{self.cap_h} @ {self.cap_fps_prop if self.cap_fps_prop > 0 else 'N/A'} FPS. Using {self.effective_fps_for_tracker:.1f} FPS for tracker.") # Logging
        else:
            self.cap_w, self.cap_h, self.cap_fps_prop, self.effective_fps_for_tracker = 0,0,0,30.0
            logger.error("ObjectDetector initialized with no valid video capture.") # Logging

        self.conf_threshold_cv = self.config.yolo.conf_thresh
        self.show_info_overlay_cv = not self.config.display.no_info # Updated
        self.active_target_classes_cv = self.config.yolo.classes # Updated (already a list or None)
        
        self.cv_class_filter_options = [None, ["person"], ["car"], ["person", "car"], ["bottle", "cup"]]
        self.cv_current_filter_index = 0
        if self.active_target_classes_cv:
            # Ensure the initial list from config is findable or add it
            if self.active_target_classes_cv not in self.cv_class_filter_options:
                 self.cv_class_filter_options.append(list(self.active_target_classes_cv)) # Ensure it's a list copy
            try: self.cv_current_filter_index = self.cv_class_filter_options.index(self.active_target_classes_cv)
            except ValueError: # Should not happen if added above, but as fallback
                 self.cv_current_filter_index = len(self.cv_class_filter_options) -1 
        else: # active_target_classes_cv is None
             self.active_target_classes_cv = self.cv_class_filter_options[self.cv_current_filter_index]


        self.video_writer = None
        self.recording_to_file = False
        self.prev_time_calc = time.time() 
        self.fps_smooth_calc = 0.0
        
        self.img_output_dir = os.path.join(self.config.app.output_dir, "images") # Updated
        self.video_output_dir = os.path.join(self.config.app.output_dir, "videos") # Updated
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        # Direct use of config for DeepFace settings
        self.emotion_detection_enabled = self.config.deepface.enable_emotion
        self.age_gender_detection_enabled = self.config.deepface.enable_age_gender
        # self.facial_analysis_detector_backend = self.config.deepface.deepface_backend # Used directly in _perform_facial_analysis
        # self.facial_analysis_interval = self.config.deepface.facial_analysis_interval # Used directly
        # self.max_faces_to_analyze_deepface = self.config.deepface.max_faces_to_analyze # Used directly

        self.frame_counter_facial_analysis = 0
        self.last_facial_analysis_results_raw = [] 

        # self.async_deepface_enabled = self.config.deepface.async_deepface # Used directly
        # self.crop_for_deepface = self.config.deepface.crop_for_deepface # Used directly
        self.deepface_thread = None
        self.deepface_input_queue = None
        self.deepface_output_queue = None
        self.deepface_stop_event = None
        
        self.tracker_type = self.config.tracker.tracker_type # Updated
        self.person_tracker = None
        if self.tracker_type == 'bytetrack' and BYTETRACK_AVAILABLE and BYTETracker is not None:
            bt_params = self.config.tracker.bytetrack_params
            logger.info(f"Initializing ByteTrack: track_thresh={bt_params.bytetrack_track_thresh}, track_buffer(frames)={bt_params.bytetrack_track_buffer}, match_thresh={bt_params.bytetrack_match_thresh}, frame_rate={self.effective_fps_for_tracker}") # Logging
            
            byte_tracker_args_ns = argparse.Namespace(
                track_thresh=bt_params.bytetrack_track_thresh,
                track_buffer=int(bt_params.bytetrack_track_buffer), 
                match_thresh=bt_params.bytetrack_match_thresh,
                mot20=False,  
            )
            self.person_tracker = BYTETracker(
                args=byte_tracker_args_ns, 
                frame_rate=self.effective_fps_for_tracker
            )
            logger.info("ByteTrack initialized.") # Logging
        else:
            if self.tracker_type == 'bytetrack' and (not BYTETRACK_AVAILABLE or BYTETracker is None):
                logger.warning("ByteTrack selected but not available/imported. Falling back to SORT.") # Logging
            self.tracker_type = 'sort' 
            sort_params = self.config.tracker.sort_params
            logger.info(f"Initializing SORT tracker: max_age={sort_params.sort_max_age}, min_hits={sort_params.sort_min_hits}, iou_thresh={sort_params.sort_iou_thresh}") # Logging
            self.person_tracker = Sort(max_age=sort_params.sort_max_age, min_hits=sort_params.sort_min_hits, iou_threshold=sort_params.sort_iou_thresh)
            logger.info("SORT tracker initialized.") # Logging
        
        self.tracked_persons_current_frame_data = {} 

        if self.config.deepface.async_deepface and (self.config.deepface.enable_emotion or self.config.deepface.enable_age_gender):
            logger.info("Initializing Asynchronous DeepFace Processing.") # Logging
            self.deepface_input_queue = queue.Queue(maxsize=1)
            self.deepface_output_queue = queue.Queue(maxsize=1)
            self.deepface_stop_event = threading.Event()
            self.deepface_thread = threading.Thread(target=self._deepface_worker, daemon=True)
            self.deepface_thread.start()

        self._print_opencv_instructions()
        self._warmup_models()
    
    def _warmup_models(self):
        model_path = self.config.yolo.model # Updated
        device_for_warmup = 'cpu' 
        # Check actual device of loaded YOLO model if possible
        if hasattr(self.yolo_model, 'device') and self.yolo_model.device is not None and str(self.yolo_model.device.type) != 'cpu':
            device_for_warmup = str(self.yolo_model.device.type)
        # Fallback to config device if model doesn't report, ensuring CUDA check
        elif self.config.yolo.device == 'cuda' and torch.cuda.is_available():
            device_for_warmup = 'cuda'
        
        if device_for_warmup != 'cpu':
            try:
                logger.info(f"Warming up YOLO model '{os.path.basename(model_path)}' on {device_for_warmup}...") # Logging
                dummy_input_yolo = torch.zeros(1, 3, 320, 320).to(torch.device(device_for_warmup))
                self.yolo_model.predict(dummy_input_yolo, verbose=False) 
                logger.info(f"YOLO model '{os.path.basename(model_path)}' warmed up.") # Logging
            except Exception as e_warmup:
                logger.warning(f"YOLO model warmup failed: {e_warmup}", exc_info=True) # Logging

        if not self.config.deepface.async_deepface and (self.config.deepface.enable_emotion or self.config.deepface.enable_age_gender):
            try:
                logger.info("Warming up DeepFace models (sync mode)...") # Logging
                dummy_frame_deepface = np.zeros((100, 100, 3), dtype=np.uint8)
                actions_warmup = []
                if self.config.deepface.enable_emotion: actions_warmup.append('emotion')
                if self.config.deepface.enable_age_gender: actions_warmup.extend(['age', 'gender'])
                # Remove duplicates if both age and gender are added (though extend handles individual items)
                actions_warmup = list(set(actions_warmup)) 
                
                if actions_warmup:
                    DeepFace.analyze(img_path=dummy_frame_deepface, actions=actions_warmup, 
                                     detector_backend=self.config.deepface.deepface_backend, 
                                     enforce_detection=False, silent=True)
                logger.info("DeepFace models warmed up (sync mode).") # Logging
            except Exception as e_df_warmup:
                logger.warning(f"DeepFace warmup (sync mode) failed: {e_df_warmup}", exc_info=True) # Logging
        elif self.config.deepface.async_deepface and (self.config.deepface.enable_emotion or self.config.deepface.enable_age_gender):
            logger.info("DeepFace warmup will occur in its dedicated thread on first analysis.") # Logging

    def _load_yolo_model(self):
        model_path = self.config.yolo.model
        device_to_use = self.config.yolo.device
        model = None
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model file not found at path: {model_path}")

            if model_path.endswith(".engine"):
                logger.info(f"Loading YOLO TensorRT engine '{os.path.basename(model_path)}'...")
                model = YOLO(model_path, task='detect')
                logger.info(f"YOLO TensorRT engine '{os.path.basename(model_path)}' loaded.")
                if hasattr(model, 'device') and model.device is not None: 
                    logger.info(f"Engine reporting device: {model.device}")
                else: 
                    logger.info("Engine device not directly queryable. Assumed to use GPU it was built for.")
            elif model_path.endswith(".pt"):
                logger.info(f"Loading YOLO PyTorch model '{os.path.basename(model_path)}' on device '{device_to_use}'...")
                model = YOLO(model_path)
                actual_device_for_pt = device_to_use
                if device_to_use == 'cuda' and not torch.cuda.is_available():
                    logger.warning(f"CUDA selected for '{os.path.basename(model_path)}' but not available. Falling back to CPU.")
                    actual_device_for_pt = 'cpu'
                model.to(actual_device_for_pt)
                
                final_model_device_type = 'unknown'
                try:
                    if hasattr(model, 'device') and model.device is not None: final_model_device_type = str(model.device.type)
                    elif hasattr(model, 'model') and hasattr(model.model, 'device') and model.model.device is not None: final_model_device_type = str(model.model.device.type)
                    elif hasattr(model, 'model') and next(model.model.parameters(), None) is not None: final_model_device_type = str(next(model.model.parameters()).device.type)
                except Exception as e_dc: 
                    logger.warning(f"Could not definitively determine .pt model's device type after loading: {e_dc}", exc_info=True)
                logger.info(f"YOLO PyTorch Model '{os.path.basename(model_path)}' loaded. Attempted device: '{actual_device_for_pt}', Reported/Final device: {final_model_device_type}.")
            else:
                raise ValueError(f"Unsupported model file type: {model_path}. Please use .pt or .engine")
            return model
        except FileNotFoundError as e:
            logger.error(f"YOLO model file not found: {e}", exc_info=True)
            raise SystemExit(f"Critical error: YOLO model file not found at {model_path}.")
        except ValueError as e: # For unsupported file type
            logger.error(f"Error loading YOLO model: {e}", exc_info=True)
            raise SystemExit(f"Critical error: Unsupported YOLO model file type {model_path}.")
        except RuntimeError as e: # Catch PyTorch/Ultralytics runtime errors
            logger.error(f"Runtime error loading YOLO model '{model_path}': {e}", exc_info=True)
            raise SystemExit(f"Critical error: Runtime error loading YOLO model {model_path}.")
        except Exception as e: # Catch-all for any other unexpected errors
            logger.error(f"An unexpected error occurred while loading YOLO model '{model_path}': {e}", exc_info=True)
            raise SystemExit(f"Critical error: Unexpected error loading YOLO model {model_path}.")


    def _init_video_capture(self):
        capture_source_str = str(self.config.app.source)
        cap = None
        try:
            try: # Try to convert to int for webcam ID
                capture_source_int = int(capture_source_str)
                logger.info(f"Attempting to initialize webcam ID: {capture_source_int}")
                cap = cv2.VideoCapture(capture_source_int)
                self.source_is_file = False
            except ValueError: # Not an int, assume it's a file path or URL
                logger.info(f"Initializing video file/URL: {capture_source_str}")
                if not (capture_source_str.startswith("http://") or capture_source_str.startswith("https://") or os.path.exists(capture_source_str)):
                    logger.error(f"Video file/URL not found or inaccessible: {capture_source_str}")
                    return None
                cap = cv2.VideoCapture(capture_source_str)
                self.source_is_file = True
            
            if not cap or not cap.isOpened():
                logger.error(f"Cannot open video source: {self.config.app.source}")
                return None
            logger.info(f"Video source successfully opened: {self.config.app.source}")
            return cap
        except cv2.error as e:
            logger.error(f"OpenCV error initializing video capture for '{self.config.app.source}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error initializing video capture for '{self.config.app.source}': {e}", exc_info=True)
            return None

    def _print_opencv_instructions(self):
        print("\n[INFO] OpenCV detection window starting...")
        print("  Press 'q' or ESC to quit."); print("  Press 's' for screenshot, 'r' to record.")
        print("  Press '+/-' for YOLO confidence."); print("  Press 'i' for info overlay, 'f' for YOLO class filter.")
        # Instructions reflect config names if they were to be displayed, but toggles are direct.
        print(f"  Press 'e' to toggle Emotion (current: {'ON' if self.config.deepface.enable_emotion else 'OFF'}).")
        print(f"  Press 'a' to toggle Age/Gender (current: {'ON' if self.config.deepface.enable_age_gender else 'OFF'}).")
        print(f"  Press 't' to toggle Person Track IDs (current: {'ON' if self.config.tracker.show_track_id else 'OFF'}).")
        print("  Window 'X' button should also close.")

    def _perform_facial_analysis_on_image(self, image_to_analyze):
        # This function is often called rapidly, keep logging minimal or at DEBUG if necessary.
        actions = []
        if self.config.deepface.enable_emotion: actions.append('emotion') # Updated
        if self.config.deepface.enable_age_gender: # Updated
            if 'age' not in actions: actions.append('age') # Redundant with extend below, but harmless
            if 'gender' not in actions: actions.append('gender')
            # actions.extend(['age', 'gender']) # More concise
        actions = list(set(actions)) # Ensure unique
        if not actions: return []
        
        try:
            raw_res = DeepFace.analyze(img_path=image_to_analyze, actions=actions, 
                                       detector_backend=self.config.deepface.deepface_backend, # Updated
                                       enforce_detection=False, silent=True)
            outputs = []
            if isinstance(raw_res, list) and raw_res:
                proc_res = raw_res
                # Use config for max_faces_to_analyze
                is_large_image = image_to_analyze.shape[0] > 200 and image_to_analyze.shape[1] > 200 # Heuristic, keep or remove
                if is_large_image and self.config.deepface.max_faces_to_analyze > 0 and len(raw_res) > self.config.deepface.max_faces_to_analyze: # Updated
                    def get_area(f): return f['region'].get('w',0)*f['region'].get('h',0) if isinstance(f,dict) and 'region' in f else 0
                    raw_res.sort(key=get_area, reverse=True); proc_res = raw_res[:self.config.deepface.max_faces_to_analyze] # Updated
                
                for face_data in proc_res:
                    if isinstance(face_data, dict) and 'region' in face_data:
                        region=face_data['region']; box=(region['x'],region['y'],region['x']+region['w'],region['y']+region['h']);
                        output={"face_box_deepface":box}
                        if 'dominant_emotion' in face_data: output["emotion"]=face_data['dominant_emotion']; output["emotion_confidence"]=face_data.get('emotion',{}).get(face_data['dominant_emotion'],0)
                        if 'age' in face_data: output["age"]=face_data['age'] 
                        if 'dominant_gender' in face_data: output["gender"]=face_data['dominant_gender']; output["gender_confidence"]=face_data.get('gender',{}).get(face_data['dominant_gender'],0)
                        outputs.append(output)
            return outputs
        except Exception as e:
            logger.warning(f"Facial analysis failed for an image: {e}", exc_info=True) # Changed to warning
            return []

    def _deepface_worker(self):
        df_worker_logger = logging.getLogger(__name__ + ".DeepFaceWorker") # Specific logger
        df_worker_logger.info("DeepFace worker thread started.") # Logging
        is_first_run = True
        while not self.deepface_stop_event.is_set():
            try:
                try: input_data = self.deepface_input_queue.get(timeout=0.5)
                except queue.Empty: continue
                frame_copy_for_analysis, tracked_persons_boxes_for_ts = input_data
                all_facial_results = []
                if is_first_run: df_worker_logger.info("DeepFace worker performing initial analysis (warmup)..."); is_first_run = False # Logging

                if self.config.deepface.crop_for_deepface and tracked_persons_boxes_for_ts:
                    persons_to_analyze = list(tracked_persons_boxes_for_ts.items())
                    if self.config.deepface.max_faces_to_analyze > 0 and len(tracked_persons_boxes_for_ts) > self.config.deepface.max_faces_to_analyze: # Updated
                        persons_to_analyze = sorted(tracked_persons_boxes_for_ts.items(), key=lambda item: (item[1][2]-item[1][0]) * (item[1][3]-item[1][1]), reverse=True)[:self.config.deepface.max_faces_to_analyze] # Updated
                    for track_id, box in persons_to_analyze:
                        x1,y1,x2,y2=box; pad_x,pad_y=int((x2-x1)*0.15),int((y2-y1)*0.20);
                        crop_x1,crop_y1=max(0,x1-pad_x),max(0,y1-pad_y); crop_x2,crop_y2=min(frame_copy_for_analysis.shape[1],x2+pad_x),min(frame_copy_for_analysis.shape[0],y2+pad_y)
                        person_crop=frame_copy_for_analysis[crop_y1:crop_y2,crop_x1:crop_x2]
                        if person_crop.size==0: continue
                        crop_results=self._perform_facial_analysis_on_image(person_crop)
                        for res in crop_results: 
                            fb_local=res["face_box_deepface"]; res["face_box_deepface"]=(fb_local[0]+crop_x1,fb_local[1]+crop_y1,fb_local[2]+crop_x1,fb_local[3]+crop_y1); res["original_track_id_hint"]=track_id 
                        all_facial_results.extend(crop_results)
                else: all_facial_results = self._perform_facial_analysis_on_image(frame_copy_for_analysis)
                
                try: self.deepface_output_queue.get_nowait()
                except queue.Empty: pass
                self.deepface_output_queue.put_nowait(all_facial_results)
            except Exception as e: df_worker_logger.error(f"Error in DeepFace worker loop: {e}", exc_info=True); time.sleep(0.5) # Logging
        df_worker_logger.info("DeepFace worker thread stopped.") # Logging

    def _process_frame_for_cv(self, frame):
        # Logging for this can be very verbose, use DEBUG if needed for yolo_target_device.
        yolo_target_device = self.config.yolo.device
        if self.config.yolo.model.endswith(".pt"): # For .pt, respect config and check availability
            if yolo_target_device == 'cuda' and not torch.cuda.is_available():
                yolo_target_device = 'cpu'
        # For .engine, device is often implicit. YOLO might handle it or use its built-for device.
        # We can pass the config device, and YOLO internally might ignore if it's an engine.
        
        yolo_results = self.yolo_model.predict(frame, conf=self.conf_threshold_cv, iou=self.config.yolo.iou_thresh, classes=self.active_target_classes_cv, verbose=False, device=yolo_target_device) # Updated iou, device
        annotated_frame = frame.copy()
        yolo_boxes_obj = yolo_results[0].boxes if yolo_results and yolo_results[0].boxes is not None else None
        
        detections_for_tracker_input = []
        if yolo_boxes_obj:
            for box_obj in yolo_boxes_obj:
                class_id=int(box_obj.cls.item()); class_name=self.yolo_model.names[class_id]
                if class_name=="person": 
                    x1,y1,x2,y2=map(int,box_obj.xyxy[0]); score=box_obj.conf.item()
                    detections_for_tracker_input.append([x1,y1,x2,y2,score])
        
        detections_np_for_tracker = np.array(detections_for_tracker_input)
        tracked_objects_list = [] 

        if self.person_tracker:
            if detections_np_for_tracker.shape[0] > 0:
                if self.tracker_type == 'bytetrack' and BYTETRACK_AVAILABLE and BYTETracker is not None:
                    online_targets = self.person_tracker.update(detections_np_for_tracker, 
                                                                (self.cap_h, self.cap_w), 
                                                                (self.cap_h, self.cap_w))
                    for t in online_targets:
                        tlwh, track_id, score = t.tlwh, t.track_id, t.score
                        if tlwh[2]*tlwh[3] > 0: 
                            tracked_objects_list.append([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3], track_id, score])
                else: 
                    sort_output = self.person_tracker.update(detections_np_for_tracker) 
                    for trk_sort in sort_output: # output is [x1,y1,x2,y2,track_id]
                        tracked_objects_list.append(list(trk_sort) + [trk_sort[4] if len(trk_sort)>4 else 1.0]) 
            else: # no detections, call update to age tracks
                empty_dets = np.empty((0,5))
                if self.tracker_type == 'bytetrack' and BYTETRACK_AVAILABLE and BYTETracker is not None:
                    online_targets = self.person_tracker.update(empty_dets, 
                                                                (self.cap_h, self.cap_w), 
                                                                (self.cap_h, self.cap_w))
                    for t in online_targets: tlwh, track_id, score = t.tlwh, t.track_id, t.score; tracked_objects_list.append([tlwh[0],tlwh[1],tlwh[0]+tlwh[2],tlwh[1]+tlwh[3],track_id,score])
                else: # SORT
                    sort_output = self.person_tracker.update(empty_dets)
                    for trk_sort in sort_output: tracked_objects_list.append(list(trk_sort) + [trk_sort[4] if len(trk_sort)>4 else 1.0])
        
        current_tracked_persons_boxes_for_deepface = {} 
        self.tracked_persons_current_frame_data = {}
        for trk_item in tracked_objects_list: 
            x1t,y1t,x2t,y2t,track_id = map(int, trk_item[:5]) 
            self.tracked_persons_current_frame_data[track_id] = {"box_yolo_sort":(x1t,y1t,x2t,y2t), "facial_analysis":None}
            current_tracked_persons_boxes_for_deepface[track_id] = (x1t,y1t,x2t,y2t)

        fa_is_active = self.config.deepface.enable_emotion or self.config.deepface.enable_age_gender # Updated
        if fa_is_active:
            self.frame_counter_facial_analysis += 1
            if self.frame_counter_facial_analysis >= self.config.deepface.facial_analysis_interval: # Updated
                self.frame_counter_facial_analysis = 0 
                if self.config.deepface.async_deepface and self.deepface_input_queue: # Updated
                    try: self.deepface_input_queue.get_nowait() 
                    except queue.Empty: pass
                    try: self.deepface_input_queue.put_nowait((frame.copy(), current_tracked_persons_boxes_for_deepface.copy()))
                    except queue.Full: pass 
                elif not self.config.deepface.async_deepface: # Updated
                    sync_results = []
                    if self.config.deepface.crop_for_deepface and current_tracked_persons_boxes_for_deepface: # Updated
                        items_to_process_sync = list(current_tracked_persons_boxes_for_deepface.items())
                        if self.config.deepface.max_faces_to_analyze > 0 and len(items_to_process_sync) > self.config.deepface.max_faces_to_analyze: # Updated
                            items_to_process_sync = sorted(items_to_process_sync,key=lambda i:(i[1][2]-i[1][0])*(i[1][3]-i[1][1]),reverse=True)[:self.config.deepface.max_faces_to_analyze] # Updated
                        for tid,box_coords in items_to_process_sync:
                            x1c,y1c,x2c,y2c=box_coords;pxc,pyc=int((x2c-x1c)*.15),int((y2c-y1c)*.2);cx1c,cy1c=max(0,x1c-pxc),max(0,y1c-pyc);cx2c,cy2c=min(frame.shape[1],x2c+pxc),min(frame.shape[0],y2c+pyc)
                            crop_img=frame[cy1c:cy2c,cx1c:cx2c];
                            if crop_img.size==0: continue
                            crop_fa_results_sync=self._perform_facial_analysis_on_image(crop_img)
                            for r_sync in crop_fa_results_sync:fb_local_sync=r_sync["face_box_deepface"];r_sync["face_box_deepface"]=(fb_local_sync[0]+cx1c,fb_local_sync[1]+cy1c,fb_local_sync[2]+cx1c,fb_local_sync[3]+cy1c)
                            sync_results.extend(crop_fa_results_sync)
                        self.last_facial_analysis_results_raw = sync_results
                    else: 
                        self.last_facial_analysis_results_raw = self._perform_facial_analysis_on_image(frame)
            
            if self.config.deepface.async_deepface and self.deepface_output_queue: # Updated
                try: self.last_facial_analysis_results_raw = self.deepface_output_queue.get_nowait()
                except queue.Empty: pass 
            if self.last_facial_analysis_results_raw and self.tracked_persons_current_frame_data:
                unmatched_fa_indices_list = list(range(len(self.last_facial_analysis_results_raw)))
                for track_id_assoc, person_data_assoc in self.tracked_persons_current_frame_data.items():
                    person_box_assoc=person_data_assoc["box_yolo_sort"]; best_iou_assoc=0; best_fa_list_idx_assoc=-1 
                    temp_unmatched_indices = unmatched_fa_indices_list[:] 
                    
                    for i_loop_idx, fa_original_idx in enumerate(temp_unmatched_indices):
                        if fa_original_idx not in unmatched_fa_indices_list : continue 
                        
                        fa_result_assoc=self.last_facial_analysis_results_raw[fa_original_idx]; fa_box_assoc=fa_result_assoc["face_box_deepface"]
                        iou_val = iou_calc_for_association(np.array(person_box_assoc), np.array(fa_box_assoc))
                        if iou_val > self.config.deepface.face_person_iou_thresh and iou_val > best_iou_assoc: # Updated
                            best_iou_assoc=iou_val; best_fa_list_idx_assoc = unmatched_fa_indices_list.index(fa_original_idx) 

                    if best_fa_list_idx_assoc != -1: 
                        matched_fa_original_idx_for_removal = unmatched_fa_indices_list.pop(best_fa_list_idx_assoc)
                        person_data_assoc["facial_analysis"]=self.last_facial_analysis_results_raw[matched_fa_original_idx_for_removal]
        
        # Updated draw_detections call arguments
        annotated_frame, yolo_detections_count_dict = draw_detections(
            annotated_frame, yolo_boxes_obj, self.yolo_model.names, 
            self.config.display.hide_labels, self.config.display.hide_conf, self.config.display.line_thickness, 
            tracked_persons_data=self.tracked_persons_current_frame_data if self.config.tracker.show_track_id else None, 
            show_person_ids = self.config.tracker.show_track_id
        )
        return annotated_frame, yolo_detections_count_dict

    def _handle_cv_keys(self, key, annotated_frame):
        if key == ord('q') or key == 27: print("[INFO] Quit key pressed."); return False
        elif key == ord('s'): timestamp=time.strftime("%Y%m%d-%H%M%S"); fn=os.path.join(self.img_output_dir,f"cv_frame_{timestamp}.jpg"); cv2.imwrite(fn,annotated_frame); print(f"[INFO] Saved to {fn}")
        elif key == ord('r'):
            self.recording_to_file = not self.recording_to_file
            if self.recording_to_file:
                if not self.video_writer: ts=time.strftime("%Y%m%d-%H%M%S"); fn=os.path.join(self.video_output_dir,f"cv_rec_{ts}.mp4"); fourcc=cv2.VideoWriter_fourcc(*'mp4v'); fps_rec=self.cap_fps_prop if self.cap_fps_prop>0 else (self.fps_smooth_calc if self.fps_smooth_calc > 0 else 20.0); self.video_writer=cv2.VideoWriter(fn,fourcc,fps_rec,(self.cap_w,self.cap_h))
                print("[INFO] Recording started.")
            else:
                if self.video_writer: self.video_writer.release(); self.video_writer=None; print("[INFO] Recording stopped.")
        elif key == ord('+') or key == ord('='): self.conf_threshold_cv=min(1.0,self.conf_threshold_cv+0.05); self.config.yolo.conf_thresh = self.conf_threshold_cv; print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}") # Update config
        elif key == ord('-') or key == ord('_'): self.conf_threshold_cv=max(0.0,self.conf_threshold_cv-0.05); self.config.yolo.conf_thresh = self.conf_threshold_cv; print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}") # Update config
        elif key == ord('i'): self.show_info_overlay_cv=not self.show_info_overlay_cv; self.config.display.no_info = not self.show_info_overlay_cv; print(f"[INFO] CV Info {'shown' if self.show_info_overlay_cv else 'hidden'}.") # Update config
        elif key == ord('f'):
            self.cv_current_filter_index=(self.cv_current_filter_index+1)%len(self.cv_class_filter_options); self.active_target_classes_cv=self.cv_class_filter_options[self.cv_current_filter_index]
            self.config.yolo.classes = self.active_target_classes_cv # Update config
            f_str="All" if not self.active_target_classes_cv else ', '.join(self.active_target_classes_cv); print(f"[INFO] CV Class filter: {f_str}")
        elif key == ord('e'):
            # Toggle based on initial launch config, then update current state and config
            self.config.deepface.enable_emotion = not self.config.deepface.enable_emotion
            self.emotion_detection_enabled = self.config.deepface.enable_emotion # Keep local toggle in sync for clarity
            print(f"[INFO] Emotion detection {'EN' if self.emotion_detection_enabled else 'DIS'}ABLED.")
            if not self.emotion_detection_enabled and not self.age_gender_detection_enabled: self.last_facial_analysis_results_raw.clear()
        elif key == ord('a'):
            self.config.deepface.enable_age_gender = not self.config.deepface.enable_age_gender
            self.age_gender_detection_enabled = self.config.deepface.enable_age_gender
            print(f"[INFO] Age/Gender detection {'EN' if self.age_gender_detection_enabled else 'DIS'}ABLED.")
            if not self.emotion_detection_enabled and not self.age_gender_detection_enabled: self.last_facial_analysis_results_raw.clear()
        elif key == ord('t'): self.config.tracker.show_track_id=not self.config.tracker.show_track_id; print(f"[INFO] Person Track IDs {'shown' if self.config.tracker.show_track_id else 'hidden'}.") # Update config
        return True

    def run_opencv_window(self):
        if not self.cap: 
            logger.error("No video capture source available to run OpenCV window.")
            return
            
        self.prev_time_calc = time.time()
        self.frame_counter_facial_analysis = 0
        self.last_facial_analysis_results_raw = []
        
        try:
            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.opencv_window_name, 1280, 720)
        except cv2.error as e:
            logger.error(f"OpenCV error setting up window: {e}", exc_info=True)
            return # Cannot proceed if window setup fails

        continue_processing = True
        while continue_processing:
            try: 
                if cv2.getWindowProperty(self.opencv_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("OpenCV window closed via 'X' button.")
                    break
            except cv2.error:
                logger.warning("OpenCV window no longer accessible (cv2.error during getWindowProperty).", exc_info=True)
                break # Exit loop if window is gone

            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video stream or failed to grab frame.")
                    break
            except cv2.error as e:
                logger.error(f"OpenCV error reading frame: {e}", exc_info=True)
                break # Cannot continue if frame reading fails
            except Exception as e:
                logger.error(f"Unexpected error reading frame: {e}", exc_info=True)
                break

            current_time = time.time()
            delta_time = current_time - self.prev_time_calc
            if delta_time > 0:
                self.fps_smooth_calc = 0.9 * self.fps_smooth_calc + 0.1 * (1.0 / delta_time)
            self.prev_time_calc = current_time
            
            try:
                annotated_frame, yolo_counts = self._process_frame_for_cv(frame)
            except Exception as e:
                logger.error(f"Error processing frame in _process_frame_for_cv: {e}", exc_info=True)
                annotated_frame = frame.copy() # Show raw frame if processing fails
                cv2.putText(annotated_frame, "FRAME PROCESSING ERROR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                yolo_counts = {} # Empty counts

            if self.show_info_overlay_cv:
                model_type_str = f" ({'PT' if self.config.yolo.model.endswith('.pt') else 'TRT'})"
                info_elements = {
                    "FPS": f"{self.fps_smooth_calc:.2f}", "Model": os.path.basename(self.config.yolo.model)+model_type_str,
                    "Tracker": self.tracker_type.upper(), "Conf": f"{self.conf_threshold_cv:.2f} (+/-)", # self.conf_threshold_cv is sync'd with config.yolo.conf_thresh
                    "Filter(f)": "All" if not self.active_target_classes_cv else ','.join(self.active_target_classes_cv), # self.active_target_classes_cv is sync'd
                    "TrackIDs(t)": "ON" if self.config.tracker.show_track_id else "OFF" # Directly use config
                }
                # Use current state for FA toggles (emotion_detection_enabled, etc.) which are sync'd with config
                fa_on = self.emotion_detection_enabled or self.age_gender_detection_enabled
                if self.config.deepface.enable_emotion is not None: # Check if original config had it, to decide if we show the toggle info
                     info_elements["Emotion(e)"] = "ON" if self.emotion_detection_enabled else "OFF"
                if self.config.deepface.enable_age_gender is not None:
                     info_elements["Age/Gen(a)"] = "ON" if self.age_gender_detection_enabled else "OFF"
                
                if fa_on: # Show FA details if any FA is active
                    info_elements["FA Interval"]=f"{self.config.deepface.facial_analysis_interval}f"; 
                    info_elements["FA Max Faces"]=f"{self.config.deepface.max_faces_to_analyze if self.config.deepface.max_faces_to_analyze > 0 else 'All'}"; 
                    info_elements["FA Async"] = "ON" if self.config.deepface.async_deepface else "OFF"; 
                    info_elements["FA Crop"] = "ON" if self.config.deepface.crop_for_deepface and fa_on else "OFF" # Crop shown if active and relevant
                annotated_frame = draw_info_overlay(annotated_frame, info_elements, self.recording_to_file, self.cap_w, yolo_counts)
            
            cv2.imshow(self.opencv_window_name, annotated_frame)
            if self.recording_to_file and self.video_writer: self.video_writer.write(annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_cv_keys(key, annotated_frame): continue_processing = False
        
        self.cleanup_opencv_window()

    def cleanup_opencv_window(self):
        logger.info("Cleaning up resources...") # Logging
        if self.config.deepface.async_deepface and self.deepface_stop_event:
            logger.info("Stopping DeepFace worker...") # Logging
            self.deepface_stop_event.set()
            if self.deepface_thread and self.deepface_thread.is_alive():
                self.deepface_thread.join(timeout=2.5)
                if self.deepface_thread.is_alive(): logger.warning("DF Thread did not stop in time.") # Logging
        if self.cap: self.cap.release(); self.cap=None
        if self.video_writer: self.video_writer.release(); self.video_writer=None
        try:
            if cv2.getWindowProperty(self.opencv_window_name, cv2.WND_PROP_VISIBLE) >= 0: # Check if window exists
                 cv2.destroyWindow(self.opencv_window_name)
        except cv2.error: 
            logger.debug("OpenCV window was already destroyed or unavailable during cleanup.", exc_info=False) # Debug
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed
        if os.name == 'posix': cv2.waitKey(50) # waitKey on POSIX can help process GUI events
        logger.info("Resources cleaned up.") # Logging