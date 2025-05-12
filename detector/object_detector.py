# detector/object_detector.py
import cv2
import time
import os
from ultralytics import YOLO
from deepface import DeepFace
from .utils import draw_detections, draw_info_overlay
import torch
import numpy as np
from .sort_tracker import Sort # KalmanBoxTracker not directly needed here anymore

class ObjectDetector:
    def __init__(self, args):
        self.args = args
        self.yolo_model = self._load_yolo_model() # This will now handle .pt or .engine
        self.cap = self._init_video_capture()
        
        if self.cap:
            self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap_fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Video source opened: {self.cap_w}x{self.cap_h} @ {self.cap_fps_prop if self.cap_fps_prop > 0 else 'N/A'} FPS")
        else:
            self.cap_w, self.cap_h, self.cap_fps_prop = 0,0,0
            print("[ERROR] ObjectDetector initialized with no valid video capture.")

        # ... (OpenCV window state, facial analysis state - same as previous version) ...
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
        self.prev_time_calc = 0 
        self.fps_smooth_calc = 0.0
        
        self.img_output_dir = os.path.join(args.output_dir, "images")
        self.video_output_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        self.emotion_detection_enabled = args.enable_emotion
        self.age_gender_detection_enabled = args.enable_age_gender
        self.facial_analysis_detector_backend = args.deepface_backend
        self.facial_analysis_interval = 5 
        self.frame_counter_facial_analysis = 0
        self.last_facial_analysis_results = [] 
        self.person_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.tracked_persons_current_frame = {} 

        self._print_opencv_instructions()


    def _load_yolo_model(self):
        model_path = self.args.model # This can be .pt or .engine
        device_to_use = self.args.device

        try:
            if model_path.endswith(".engine"):
                print(f"[INFO] Loading YOLO TensorRT engine '{model_path}'...")
                # For TensorRT engines, Ultralytics YOLO automatically uses the GPU the engine was built for/on.
                # The 'device' argument to YOLO() might be ignored or used differently for engines.
                # It's generally safer to assume the engine dictates its device, or TRT runtime picks one.
                model = YOLO(model_path, task='detect') # Specify task if not clear from engine name context
                print(f"[INFO] YOLO TensorRT engine '{model_path}' loaded.")
                # To confirm device, one might try a dummy predict and check output device,
                # but Ultralytics abstracts much of the TRT runtime details.
                # If model.device exists and shows 'cuda', great.
                if hasattr(model, 'device'):
                     print(f"[INFO] Engine reporting device: {model.device}")
                else:
                     print("[INFO] Engine device not directly queryable via model.device (normal for TRT engines via Ultralytics).")

            elif model_path.endswith(".pt"):
                print(f"[INFO] Loading YOLO PyTorch model '{model_path}' on device '{device_to_use}'...")
                model = YOLO(model_path)
                model.to(device_to_use)
                # ... (your existing .pt model device verification logic) ...
                actual_device_type = "unknown"
                try:
                    if device_to_use == 'cpu': actual_device_type = 'cpu'
                    elif torch.cuda.is_available() and device_to_use == 'cuda': 
                        dummy_input_tensor = torch.zeros(1, 3, 32, 32, device=device_to_use)
                        with torch.no_grad(): model(dummy_input_tensor) 
                        if hasattr(model, 'model') and next(model.model.parameters(), None) is not None: actual_device_type = next(model.model.parameters()).device.type
                        elif hasattr(model, 'device') and model.device is not None: actual_device_type = model.device.type
                    else: 
                        if hasattr(model, 'device') and model.device is not None: actual_device_type = model.device.type
                        else: actual_device_type = device_to_use
                except Exception as e_dc: print(f"[WARNING] Could not definitively determine .pt model's device via dummy input: {e_dc}")
                if actual_device_type == 'cuda': print(f"[INFO] YOLO PyTorch Model '{model_path}' confirmed on GPU (CUDA).")
                else: print(f"[INFO] YOLO PyTorch Model '{model_path}' loaded. Reported device type: {actual_device_type}.")

            else:
                raise ValueError(f"Unsupported model file type: {model_path}. Please use .pt or .engine")
            
            return model
            
        except Exception as e:
            print(f"[ERROR] Could not load YOLO model/engine '{model_path}': {e}")
            import traceback
            traceback.print_exc()
            raise SystemExit

    # ... (_init_video_capture, _print_opencv_instructions, _perform_facial_analysis - same as previous) ...
    def _init_video_capture(self):
        capture_source_str = str(self.args.source) 
        try:
            capture_source_int = int(capture_source_str)
            print(f"[INFO] Initializing webcam with ID: {capture_source_int}")
            cap = cv2.VideoCapture(capture_source_int)
            self.source_is_file = False
        except ValueError:
            print(f"[INFO] Initializing video file: {capture_source_str}")
            if not os.path.exists(capture_source_str):
                print(f"[ERROR] Video file not found: {capture_source_str}")
                return None
            cap = cv2.VideoCapture(capture_source_str)
            self.source_is_file = True
        if not cap or not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.args.source}")
            return None
        return cap

    def _print_opencv_instructions(self):
        print("\n[INFO] OpenCV detection window starting...")
        print("  Press 'q' or ESC to quit.")
        print("  Press 's' for screenshot, 'r' to record.")
        print("  Press '+/-' for YOLO confidence.")
        print("  Press 'i' for info overlay, 'f' for YOLO class filter.")
        print("  Press 'e' to toggle Emotion detection (if enabled at launch).")
        print("  Press 'a' to toggle Age/Gender detection (if enabled at launch).")

    def _perform_facial_analysis(self, frame_to_analyze):
        actions_to_perform = []
        if self.emotion_detection_enabled: actions_to_perform.append('emotion')
        if self.age_gender_detection_enabled: actions_to_perform.extend(['age', 'gender'])
        if not actions_to_perform: return []
        try:
            results = DeepFace.analyze(img_path=frame_to_analyze, actions=actions_to_perform, 
                                       detector_backend=self.facial_analysis_detector_backend,
                                       enforce_detection=False, silent=True)
            analysis_outputs = []
            if isinstance(results, list):
                for face_data in results:
                    if isinstance(face_data, dict) and 'region' in face_data:
                        region = face_data['region']
                        output = {"box": (region['x'], region['y'], region['w'], region['h'])}
                        if 'dominant_emotion' in face_data:
                            output["emotion"] = face_data['dominant_emotion']
                            output["emotion_confidence"] = face_data.get('emotion', {}).get(face_data['dominant_emotion'])
                        if 'age' in face_data: output["age"] = face_data['age']
                        if 'dominant_gender' in face_data:
                            output["gender"] = face_data['dominant_gender']
                            output["gender_confidence"] = face_data.get('gender', {}).get(face_data['dominant_gender'])
                        analysis_outputs.append(output)
            return analysis_outputs
        except Exception as e: print(f"[WARNING] DeepFace analysis failed: {e}"); return []


    def _process_frame_for_cv(self, frame): # (same as previous version using SORT)
        # ...
        # Note: self.yolo_model.predict will use the loaded .pt or .engine model
        # The `device` argument to predict might be more relevant for .pt files.
        # For .engine files, TRT often manages the device internally.
        # Ultralytics' YOLO wrapper tries to abstract this.
        yolo_predict_device = self.args.device if self.args.model.endswith(".pt") else None # Tentative: be explicit for .pt

        yolo_results_obj = self.yolo_model.predict(frame, conf=self.conf_threshold_cv, iou=self.args.iou_thresh, 
                                     classes=self.active_target_classes_cv, verbose=False, device=yolo_predict_device) 
        
        annotated_frame_copy = frame.copy()
        yolo_detections = yolo_results_obj[0].boxes if yolo_results_obj and yolo_results_obj[0].boxes else None
        
        detections_for_sort = []
        if yolo_detections:
            for box in yolo_detections:
                class_id = int(box.cls.item())
                class_name = self.yolo_model.names[class_id]
                if class_name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = box.conf.item()
                    detections_for_sort.append([x1, y1, x2, y2, score])
        
        tracked_objects_sort = self.person_tracker.update(np.array(detections_for_sort))
        
        self.tracked_persons_current_frame = {}
        person_tracking_draw_info = []
        for trk in tracked_objects_sort:
            x1, y1, x2, y2, track_id = map(int, trk)
            self.tracked_persons_current_frame[track_id] = [x1, y1, x2, y2]
            person_tracking_draw_info.append({ "id": track_id, "box": (x1, y1, x2, y2) })
        
        current_facial_analysis_results_to_draw = []
        perform_analysis = self.emotion_detection_enabled or self.age_gender_detection_enabled
        if perform_analysis:
            self.frame_counter_facial_analysis += 1
            if self.frame_counter_facial_analysis % self.facial_analysis_interval == 0 or not self.last_facial_analysis_results:
                self.last_facial_analysis_results = self._perform_facial_analysis(frame) 
            current_facial_analysis_results_to_draw = self.last_facial_analysis_results
        
        annotated_frame_copy, yolo_detections_count = draw_detections(
            annotated_frame_copy, yolo_detections, self.yolo_model.names, 
            self.args.hide_labels, self.args.hide_conf, self.args.line_thickness,
            facial_analysis_results=current_facial_analysis_results_to_draw,
            tracked_persons_info=person_tracking_draw_info
        )
        return annotated_frame_copy, yolo_detections_count


    def _handle_cv_keys(self, key, annotated_frame): # (same as previous version)
        # ...
        if key == ord('q') or key == 27: return False
        elif key == ord('s'): timestamp = time.strftime("%Y%m%d-%H%M%S"); filename = os.path.join(self.img_output_dir, f"cv_frame_{timestamp}.jpg"); cv2.imwrite(filename, annotated_frame); print(f"[INFO] Saved frame to {filename}")
        elif key == ord('r'):
            self.recording_to_file = not self.recording_to_file
            if self.recording_to_file:
                if self.video_writer is None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S"); filename = os.path.join(self.video_output_dir, f"cv_rec_{timestamp}.mp4"); fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    effective_fps_rec = self.cap_fps_prop if self.cap_fps_prop > 0 else (self.fps_smooth_calc if self.fps_smooth_calc > 0 else 20.0)
                    self.video_writer = cv2.VideoWriter(filename, fourcc, effective_fps_rec, (self.cap_w, self.cap_h))
                print("[INFO] Started recording.")
            else:
                if self.video_writer: self.video_writer.release(); self.video_writer = None; print("[INFO] Stopped recording.")
        elif key == ord('+') or key == ord('='): self.conf_threshold_cv = min(1.0, self.conf_threshold_cv + 0.05); print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}")
        elif key == ord('-') or key == ord('_'): self.conf_threshold_cv = max(0.0, self.conf_threshold_cv - 0.05); print(f"[INFO] CV Conf: {self.conf_threshold_cv:.2f}")
        elif key == ord('i'): self.show_info_overlay_cv = not self.show_info_overlay_cv; print(f"[INFO] CV Info {'shown' if self.show_info_overlay_cv else 'hidden'}.")
        elif key == ord('f'):
            self.cv_current_filter_index = (self.cv_current_filter_index + 1) % len(self.cv_class_filter_options)
            self.active_target_classes_cv = self.cv_class_filter_options[self.cv_current_filter_index]
            filter_str = "All" if self.active_target_classes_cv is None else ', '.join(self.active_target_classes_cv)
            print(f"[INFO] CV Class filter: {filter_str}")
        elif key == ord('e'): 
            if self.args.enable_emotion:
                self.emotion_detection_enabled = not self.emotion_detection_enabled
                print(f"[INFO] Emotion detection {'ENABLED' if self.emotion_detection_enabled else 'DISABLED'}.")
                if not self.emotion_detection_enabled and not self.age_gender_detection_enabled: self.last_facial_analysis_results = []
            else: print("[INFO] Emotion detection not enabled at launch.")
        elif key == ord('a'):
            if self.args.enable_age_gender:
                self.age_gender_detection_enabled = not self.age_gender_detection_enabled
                print(f"[INFO] Age/Gender detection {'ENABLED' if self.age_gender_detection_enabled else 'DISABLED'}.")
                if not self.emotion_detection_enabled and not self.age_gender_detection_enabled: self.last_facial_analysis_results = []
            else: print("[INFO] Age/Gender detection not enabled at launch.")
        return True


    def run_opencv_window(self): # (same as previous version, SORT re-init is key)
        # ...
        if not self.cap: print("[ERROR] Cannot run, video capture not available."); return
        self.prev_time_calc = time.time()
        self.person_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) 
        # KalmanBoxTracker.count is reset inside Sort.__init__

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_is_file: print("[INFO] End of video file.")
                else: print("[ERROR] Failed to grab frame.")
                break
            current_time_calc = time.time()
            delta_time = current_time_calc - self.prev_time_calc
            if delta_time > 0: self.fps_smooth_calc = 0.9 * self.fps_smooth_calc + 0.1 * (1.0 / delta_time)
            self.prev_time_calc = current_time_calc

            annotated_frame, yolo_detections_count = self._process_frame_for_cv(frame)

            if self.show_info_overlay_cv:
                active_classes_str_cv = "All" if self.active_target_classes_cv is None else ', '.join(self.active_target_classes_cv)
                annotated_frame = draw_info_overlay(
                    annotated_frame, self.fps_smooth_calc, self.args.model,
                    self.conf_threshold_cv, active_classes_str_cv, 
                    self.recording_to_file, self.cap_w, yolo_detections_count
                )
            cv2.imshow("Multi-Analysis Detector (OpenCV)", annotated_frame)
            if self.recording_to_file and self.video_writer: self.video_writer.write(annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_cv_keys(key, annotated_frame): break 
        self.cleanup_opencv_window()


    def cleanup_opencv_window(self): # (same as before)
        # ...
        print("[INFO] Cleaning up OpenCV window resources...")
        if self.cap: self.cap.release(); self.cap = None
        if self.video_writer: self.video_writer.release(); self.video_writer = None
        cv2.destroyAllWindows()
        print("[INFO] OpenCV window resources cleaned up.")