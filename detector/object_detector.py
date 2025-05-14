import cv2
import time
import os
import argparse 
from ultralytics import YOLO
from deepface import DeepFace
from .utils import draw_detections, draw_info_overlay, iou_calc_for_association
import torch
import numpy as np
from .sort_tracker import Sort 

BYTETRACK_AVAILABLE = False
BYTETracker = None  

try:
    from .byte_tracker import BYTETracker 
    BYTETRACK_AVAILABLE = True
    print("[INFO] Successfully imported BYTETracker from local './byte_tracker.py'.")
except ImportError as e:
    print(f"[INFO] Failed to import BYTETracker directly (from .byte_tracker): {e}. Trying other common import paths...")
    try:
        from byte_tracker import BYTETracker 
        BYTETRACK_AVAILABLE = True
        print("[INFO] Imported ByteTrack using 'from byte_tracker import BYTETracker' (likely pip package).")
    except ImportError:
        try:
            from bytetracker import BYTETracker
            BYTETRACK_AVAILABLE = True
            print("[INFO] Imported ByteTrack using 'from bytetracker import BYTETracker' (likely pip package).")
        except ImportError:
            try:
                from yolox.tracker.byte_tracker import BYTETracker 
                BYTETRACK_AVAILABLE = True
                print("[INFO] Imported ByteTrack using 'from yolox.tracker.byte_tracker import BYTETracker'.")
            except ImportError:
                print("[WARNING] ByteTrack not found via common import paths. Ensure manual files (byte_tracker.py, basetrack.py, kalman_filter.py, matching.py) are in 'detector/' and use relative imports, or install a compatible package ('yolox' or 'bytetracker').")
                print("          Falling back to SORT if ByteTrack is selected as tracker type.")


import threading
import queue

class ObjectDetector:
    def __init__(self, args):
        self.args = args
        self.yolo_model = self._load_yolo_model()
        self.cap = self._init_video_capture()
        self.opencv_window_name = "Multi-Analysis Detector (OpenCV)"
        
        if self.cap:
            self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap_fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
            self.effective_fps_for_tracker = self.cap_fps_prop if self.cap_fps_prop > 0 else 30.0
            print(f"[INFO] Video source opened: {self.cap_w}x{self.cap_h} @ {self.cap_fps_prop if self.cap_fps_prop > 0 else 'N/A'} FPS. Using {self.effective_fps_for_tracker:.1f} FPS for tracker.")
        else:
            self.cap_w, self.cap_h, self.cap_fps_prop, self.effective_fps_for_tracker = 0,0,0,30.0
            print("[ERROR] ObjectDetector initialized with no valid video capture.")

        self.conf_threshold_cv = args.conf_thresh
        self.show_info_overlay_cv = not args.no_info
        self.active_target_classes_cv = args.classes.split(',') if args.classes else None
        self.cv_class_filter_options = [None, ["person"], ["car"], ["person", "car"], ["bottle", "cup"]]
        self.cv_current_filter_index = 0
        if self.active_target_classes_cv:
            try: self.cv_current_filter_index = self.cv_class_filter_options.index(self.active_target_classes_cv)
            except ValueError: self.cv_class_filter_options.append(self.active_target_classes_cv); self.cv_current_filter_index = len(self.cv_class_filter_options) - 1
        else: self.active_target_classes_cv = self.cv_class_filter_options[self.cv_current_filter_index]

        self.video_writer = None
        self.recording_to_file = False
        self.prev_time_calc = time.time() 
        self.fps_smooth_calc = 0.0
        
        self.img_output_dir = os.path.join(args.output_dir, "images")
        self.video_output_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        self.emotion_detection_enabled = args.enable_emotion
        self.age_gender_detection_enabled = args.enable_age_gender
        self.facial_analysis_detector_backend = args.deepface_backend
        self.facial_analysis_interval = args.facial_analysis_interval
        self.max_faces_to_analyze_deepface = args.max_faces_to_analyze
        self.frame_counter_facial_analysis = 0
        self.last_facial_analysis_results_raw = [] 

        self.async_deepface_enabled = args.async_deepface
        self.crop_for_deepface = args.crop_for_deepface
        self.deepface_thread = None
        self.deepface_input_queue = None
        self.deepface_output_queue = None
        self.deepface_stop_event = None
        
        self.tracker_type = args.tracker_type
        self.person_tracker = None
        if self.tracker_type == 'bytetrack' and BYTETRACK_AVAILABLE and BYTETracker is not None:
            print(f"[INFO] Initializing ByteTrack: track_thresh={args.bytetrack_track_thresh}, track_buffer(frames)={args.bytetrack_track_buffer}, match_thresh={args.bytetrack_match_thresh}, frame_rate={self.effective_fps_for_tracker}")
            
            byte_tracker_args = argparse.Namespace(
                track_thresh=args.bytetrack_track_thresh,
                track_buffer=int(args.bytetrack_track_buffer), 
                match_thresh=args.bytetrack_match_thresh,
                mot20=False,  
            )

            self.person_tracker = BYTETracker(
                args=byte_tracker_args, 
                frame_rate=self.effective_fps_for_tracker
            )
            print("[INFO] ByteTrack initialized.")
        else:
            if self.tracker_type == 'bytetrack' and (not BYTETRACK_AVAILABLE or BYTETracker is None):
                print("[WARNING] ByteTrack selected but not available/imported. Falling back to SORT.")
            self.tracker_type = 'sort' 
            print(f"[INFO] Initializing SORT tracker: max_age={args.sort_max_age}, min_hits={args.sort_min_hits}, iou_thresh={args.sort_iou_thresh}")
            self.person_tracker = Sort(max_age=args.sort_max_age, min_hits=args.sort_min_hits, iou_threshold=args.sort_iou_thresh)
            print("[INFO] SORT tracker initialized.")
        
        self.tracked_persons_current_frame_data = {} 

        if self.async_deepface_enabled and (self.emotion_detection_enabled or self.age_gender_detection_enabled):
            print("[INFO] Initializing Asynchronous DeepFace Processing.")
            self.deepface_input_queue = queue.Queue(maxsize=1)
            self.deepface_output_queue = queue.Queue(maxsize=1)
            self.deepface_stop_event = threading.Event()
            self.deepface_thread = threading.Thread(target=self._deepface_worker, daemon=True)
            self.deepface_thread.start()

        self._print_opencv_instructions()
        self._warmup_models()
    
    def _warmup_models(self):
        model_path = self.args.model
        device_for_warmup = 'cpu' 
        if hasattr(self.yolo_model, 'device') and self.yolo_model.device is not None and str(self.yolo_model.device.type) != 'cpu':
            device_for_warmup = str(self.yolo_model.device.type)
        elif self.args.device == 'cuda' and torch.cuda.is_available():
            device_for_warmup = 'cuda'
        
        if device_for_warmup != 'cpu':
            try:
                print(f"[INFO] Warming up YOLO model '{os.path.basename(model_path)}' on {device_for_warmup}...")
                dummy_input_yolo = torch.zeros(1, 3, 320, 320).to(torch.device(device_for_warmup))
                self.yolo_model.predict(dummy_input_yolo, verbose=False)
                print(f"[INFO] YOLO model '{os.path.basename(model_path)}' warmed up.")
            except Exception as e_warmup:
                print(f"[WARNING] YOLO model warmup failed: {e_warmup}")

        if not self.async_deepface_enabled and (self.emotion_detection_enabled or self.age_gender_detection_enabled):
            try:
                print("[INFO] Warming up DeepFace models (sync mode)...")
                dummy_frame_deepface = np.zeros((100, 100, 3), dtype=np.uint8)
                actions_warmup = [a for a,flag in [('emotion',self.emotion_detection_enabled),('age',self.age_gender_detection_enabled),('gender',self.age_gender_detection_enabled)] if flag]
                if 'age' in actions_warmup and 'gender' in actions_warmup: 
                    if 'age' in actions_warmup: actions_warmup.remove('age') 
                
                if actions_warmup:
                    DeepFace.analyze(img_path=dummy_frame_deepface, actions=actions_warmup, 
                                     detector_backend=self.facial_analysis_detector_backend, 
                                     enforce_detection=False, silent=True)
                print("[INFO] DeepFace models warmed up (sync mode).")
            except Exception as e_df_warmup:
                print(f"[WARNING] DeepFace warmup (sync mode) failed: {e_df_warmup}")
        elif self.async_deepface_enabled and (self.emotion_detection_enabled or self.age_gender_detection_enabled):
            print("[INFO] DeepFace warmup will occur in its dedicated thread on first analysis.")

    def _load_yolo_model(self):
        model_path = self.args.model
        device_to_use = self.args.device
        try:
            if model_path.endswith(".engine"):
                print(f"[INFO] Loading YOLO TensorRT engine '{os.path.basename(model_path)}'...")
                model = YOLO(model_path, task='detect')
                print(f"[INFO] YOLO TensorRT engine '{os.path.basename(model_path)}' loaded.")
                if hasattr(model, 'device') and model.device is not None: print(f"[INFO] Engine reporting device: {model.device}")
                else: print("[INFO] Engine device not directly queryable. Assumed to use GPU it was built for.")
            elif model_path.endswith(".pt"):
                print(f"[INFO] Loading YOLO PyTorch model '{os.path.basename(model_path)}' on device '{device_to_use}'...")
                model = YOLO(model_path)
                actual_device = device_to_use
                if device_to_use == 'cuda' and not torch.cuda.is_available(): print(f"[WARNING] CUDA selected but not available. Falling back to CPU for {os.path.basename(model_path)}."); actual_device = 'cpu'
                model.to(actual_device) 
                actual_device_type = actual_device 
                try:
                    if hasattr(model, 'device') and model.device is not None: actual_device_type = str(model.device.type) 
                    elif hasattr(model, 'model') and next(model.model.parameters(), None) is not None: actual_device_type = str(next(model.model.parameters()).device.type)
                except Exception as e_dc: print(f"[WARNING] Could not definitively determine .pt model's device type: {e_dc}")
                print(f"[INFO] YOLO PyTorch Model '{os.path.basename(model_path)}' loaded. Reported device: {actual_device_type}.")
            else: raise ValueError(f"Unsupported model file type: {model_path}. Please use .pt or .engine")
            return model
        except Exception as e: print(f"[ERROR] Could not load YOLO model/engine '{model_path}': {e}"); import traceback; traceback.print_exc(); raise SystemExit

    def _init_video_capture(self):
        capture_source_str = str(self.args.source) 
        try:
            capture_source_int = int(capture_source_str)
            cap = cv2.VideoCapture(capture_source_int) # Add API preference if needed: cv2.VideoCapture(id, cv2.CAP_DSHOW)
            self.source_is_file = False; print(f"[INFO] Initializing webcam ID: {capture_source_int}")
        except ValueError:
            if not os.path.exists(capture_source_str): print(f"[ERROR] Video file not found: {capture_source_str}"); return None
            cap = cv2.VideoCapture(capture_source_str)
            self.source_is_file = True; print(f"[INFO] Initializing video file: {capture_source_str}")
        if not cap or not cap.isOpened(): print(f"[ERROR] Cannot open video source: {self.args.source}"); return None
        return cap

    def _print_opencv_instructions(self):
        print("\n[INFO] OpenCV detection window starting...")
        print("  Press 'q' or ESC to quit."); print("  Press 's' for screenshot, 'r' to record.")
        print("  Press '+/-' for YOLO confidence."); print("  Press 'i' for info overlay, 'f' for YOLO class filter.")
        print("  Press 'e' to toggle Emotion detection."); print("  Press 'a' to toggle Age/Gender detection.")
        print("  Press 't' to toggle Person Tracking IDs."); print("  Window 'X' button should also close.")

    def _perform_facial_analysis_on_image(self, image_to_analyze):
        actions = []
        if self.emotion_detection_enabled: actions.append('emotion')
        if self.age_gender_detection_enabled:
            if 'age' not in actions: actions.append('age')
            if 'gender' not in actions: actions.append('gender')
        if not actions: return []
        
        try:
            raw_res = DeepFace.analyze(img_path=image_to_analyze, actions=actions, 
                                       detector_backend=self.facial_analysis_detector_backend, 
                                       enforce_detection=False, silent=True)
            outputs = []
            if isinstance(raw_res, list) and raw_res:
                proc_res = raw_res
                is_large_image = image_to_analyze.shape[0] > 200 and image_to_analyze.shape[1] > 200
                if is_large_image and self.max_faces_to_analyze_deepface > 0 and len(raw_res) > self.max_faces_to_analyze_deepface:
                    def get_area(f): return f['region'].get('w',0)*f['region'].get('h',0) if isinstance(f,dict) and 'region' in f else 0
                    raw_res.sort(key=get_area, reverse=True); proc_res = raw_res[:self.max_faces_to_analyze_deepface]
                
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
            return []

    def _deepface_worker(self):
        print("[INFO] DeepFace worker thread started.")
        is_first_run = True
        while not self.deepface_stop_event.is_set():
            try:
                try: input_data = self.deepface_input_queue.get(timeout=0.5) 
                except queue.Empty: continue 
                frame_copy_for_analysis, tracked_persons_boxes_for_ts = input_data
                all_facial_results = []
                if is_first_run: print("[INFO] DeepFace worker performing initial analysis (warmup)..."); is_first_run = False

                if self.crop_for_deepface and tracked_persons_boxes_for_ts:
                    persons_to_analyze = list(tracked_persons_boxes_for_ts.items()) 
                    if self.max_faces_to_analyze_deepface > 0 and len(tracked_persons_boxes_for_ts) > self.max_faces_to_analyze_deepface:
                        persons_to_analyze = sorted(tracked_persons_boxes_for_ts.items(), key=lambda item: (item[1][2]-item[1][0]) * (item[1][3]-item[1][1]), reverse=True)[:self.max_faces_to_analyze_deepface]
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
            except Exception as e: print(f"[ERROR] DeepFace worker: {e}"); import traceback; traceback.print_exc(); time.sleep(0.5) 
        print("[INFO] DeepFace worker thread stopped.")

    def _process_frame_for_cv(self, frame):
        yolo_device = self.args.device if self.args.model.endswith(".pt") else ('cuda' if torch.cuda.is_available() else 'cpu')
        if yolo_device == 'cuda' and not torch.cuda.is_available(): yolo_device = 'cpu'
        
        yolo_results = self.yolo_model.predict(frame, conf=self.conf_threshold_cv, iou=self.args.iou_thresh, classes=self.active_target_classes_cv, verbose=False, device=yolo_device) 
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

        fa_is_active = self.emotion_detection_enabled or self.age_gender_detection_enabled
        if fa_is_active:
            self.frame_counter_facial_analysis += 1
            if self.frame_counter_facial_analysis >= self.facial_analysis_interval:
                self.frame_counter_facial_analysis = 0 
                if self.async_deepface_enabled and self.deepface_input_queue:
                    try: self.deepface_input_queue.get_nowait() 
                    except queue.Empty: pass
                    try: self.deepface_input_queue.put_nowait((frame.copy(), current_tracked_persons_boxes_for_deepface.copy()))
                    except queue.Full: pass 
                elif not self.async_deepface_enabled: 
                    sync_results = []
                    if self.crop_for_deepface and current_tracked_persons_boxes_for_deepface:
                        items_to_process_sync = list(current_tracked_persons_boxes_for_deepface.items())
                        if self.max_faces_to_analyze_deepface > 0 and len(items_to_process_sync) > self.max_faces_to_analyze_deepface:
                            items_to_process_sync = sorted(items_to_process_sync,key=lambda i:(i[1][2]-i[1][0])*(i[1][3]-i[1][1]),reverse=True)[:self.max_faces_to_analyze_deepface]
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
            
            if self.async_deepface_enabled and self.deepface_output_queue: # check for async results
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
                        if iou_val > self.args.face_person_iou_thresh and iou_val > best_iou_assoc: 
                            best_iou_assoc=iou_val; best_fa_list_idx_assoc = unmatched_fa_indices_list.index(fa_original_idx) # Get index in the original list for removal

                    if best_fa_list_idx_assoc != -1: 
                        matched_fa_original_idx_for_removal = unmatched_fa_indices_list.pop(best_fa_list_idx_assoc)
                        person_data_assoc["facial_analysis"]=self.last_facial_analysis_results_raw[matched_fa_original_idx_for_removal]
        
        annotated_frame, yolo_detections_count_dict = draw_detections(annotated_frame, yolo_boxes_obj, self.yolo_model.names, self.args.hide_labels, self.args.hide_conf, self.args.line_thickness, tracked_persons_data=self.tracked_persons_current_frame_data if self.args.show_track_id else None, show_person_ids = self.args.show_track_id)
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
        elif key == ord('+') or key == ord('='): self.conf_threshold_cv=min(1.0,self.conf_threshold_cv+0.05); print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}")
        elif key == ord('-') or key == ord('_'): self.conf_threshold_cv=max(0.0,self.conf_threshold_cv-0.05); print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}")
        elif key == ord('i'): self.show_info_overlay_cv=not self.show_info_overlay_cv; print(f"[INFO] CV Info {'shown' if self.show_info_overlay_cv else 'hidden'}.")
        elif key == ord('f'):
            self.cv_current_filter_index=(self.cv_current_filter_index+1)%len(self.cv_class_filter_options); self.active_target_classes_cv=self.cv_class_filter_options[self.cv_current_filter_index]
            f_str="All" if not self.active_target_classes_cv else ', '.join(self.active_target_classes_cv); print(f"[INFO] CV Class filter: {f_str}")
        elif key == ord('e'):
            if self.args.enable_emotion: self.emotion_detection_enabled=not self.emotion_detection_enabled; print(f"[INFO] Emotion detection {'EN' if self.emotion_detection_enabled else 'DIS'}ABLED."); L=self.last_facial_analysis_results_raw; L.clear() if not self.emotion_detection_enabled and not self.age_gender_detection_enabled else None
            else: print("[INFO] Emotion detection not enabled at launch.")
        elif key == ord('a'):
            if self.args.enable_age_gender: self.age_gender_detection_enabled=not self.age_gender_detection_enabled; print(f"[INFO] Age/Gender detection {'EN' if self.age_gender_detection_enabled else 'DIS'}ABLED."); L=self.last_facial_analysis_results_raw; L.clear() if not self.emotion_detection_enabled and not self.age_gender_detection_enabled else None
            else: print("[INFO] Age/Gender detection not enabled at launch.")
        elif key == ord('t'): self.args.show_track_id=not self.args.show_track_id; print(f"[INFO] Person Track IDs {'shown' if self.args.show_track_id else 'hidden'}.")
        return True

    def run_opencv_window(self):
        if not self.cap: print("[ERROR] No video capture."); return
        self.prev_time_calc = time.time(); self.frame_counter_facial_analysis=0; self.last_facial_analysis_results_raw=[]
        cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(self.opencv_window_name, 1280, 720) 

        continue_processing = True
        while continue_processing:
            try: # Check for 'X' button closure
                if cv2.getWindowProperty(self.opencv_window_name, cv2.WND_PROP_VISIBLE) < 1: 
                    print("[INFO] OpenCV window closed via 'X' button."); break 
            except cv2.error: print("[INFO] OpenCV window no longer accessible (cv2.error)."); break

            ret, frame = self.cap.read()
            if not ret: print("[INFO] End of video or failed grab."); break
            
            current_time = time.time(); delta_time = current_time - self.prev_time_calc
            if delta_time > 0: self.fps_smooth_calc = 0.9*self.fps_smooth_calc + 0.1*(1.0/delta_time)
            self.prev_time_calc = current_time
            
            annotated_frame, yolo_counts = self._process_frame_for_cv(frame)

            if self.show_info_overlay_cv:
                model_type_str = f" ({'PT' if self.args.model.endswith('.pt') else 'TRT'})"
                info_elements = {
                    "FPS": f"{self.fps_smooth_calc:.2f}", "Model": os.path.basename(self.args.model)+model_type_str,
                    "Tracker": self.tracker_type.upper(), "Conf": f"{self.conf_threshold_cv:.2f} (+/-)",
                    "Filter(f)": "All" if not self.active_target_classes_cv else ','.join(self.active_target_classes_cv),
                    "TrackIDs(t)": "ON" if self.args.show_track_id else "OFF"
                }
                fa_on = self.emotion_detection_enabled or self.age_gender_detection_enabled
                if self.args.enable_emotion: info_elements["Emotion(e)"] = "ON" if self.emotion_detection_enabled else "OFF"
                if self.args.enable_age_gender: info_elements["Age/Gen(a)"] = "ON" if self.age_gender_detection_enabled else "OFF"
                if fa_on: 
                    info_elements["FA Interval"]=f"{self.facial_analysis_interval}f"; 
                    info_elements["FA Max Faces"]=f"{self.max_faces_to_analyze_deepface if self.max_faces_to_analyze_deepface>0 else 'All'}"; 
                    info_elements["FA Async"] = "ON" if self.async_deepface_enabled else "OFF"; 
                    info_elements["FA Crop"] = "ON" if self.crop_for_deepface and fa_on else "OFF"
                annotated_frame = draw_info_overlay(annotated_frame, info_elements, self.recording_to_file, self.cap_w, yolo_counts)
            
            cv2.imshow(self.opencv_window_name, annotated_frame)
            if self.recording_to_file and self.video_writer: self.video_writer.write(annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_cv_keys(key, annotated_frame): continue_processing = False
        
        self.cleanup_opencv_window()

    def cleanup_opencv_window(self):
        print("[INFO] Cleaning up resources...")
        if self.async_deepface_enabled and self.deepface_stop_event:
            print("[INFO] Stopping DeepFace worker..."); self.deepface_stop_event.set()
            if self.deepface_thread and self.deepface_thread.is_alive(): 
                self.deepface_thread.join(timeout=2.5)
                if self.deepface_thread.is_alive(): print("[WARNING] DF Thread did not stop in time.")
        if self.cap: self.cap.release(); self.cap=None
        if self.video_writer: self.video_writer.release(); self.video_writer=None
        try:
            if cv2.getWindowProperty(self.opencv_window_name, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.destroyWindow(self.opencv_window_name)
        except cv2.error: pass
        cv2.destroyAllWindows() 
        if os.name == 'posix': cv2.waitKey(50) 
        print("[INFO] Resources cleaned up.")