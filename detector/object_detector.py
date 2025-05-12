# detector/object_detector.py
import cv2
import time
import os
from ultralytics import YOLO
from deepface import DeepFace # Import DeepFace
from .utils import draw_detections, draw_info_overlay # draw_detections is now modified
import torch
import numpy as np # For DeepFace input if needed

class ObjectDetector:
    def __init__(self, args):
        self.args = args
        self.yolo_model = self._load_yolo_model() # Renamed for clarity
        # Emotion model will be used directly via DeepFace.analyze
        self.cap = self._init_video_capture()
        
        if self.cap:
            self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cap_fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Video source opened: {self.cap_w}x{self.cap_h} @ {self.cap_fps_prop if self.cap_fps_prop > 0 else 'N/A'} FPS (property)")
        else:
            print("[ERROR] ObjectDetector initialized with no valid video capture.")
            self.cap_w, self.cap_h, self.cap_fps_prop = 0,0,0

        # OpenCV window interaction state
        self.conf_threshold_cv = args.conf_thresh
        self.show_info_overlay_cv = not args.no_info
        self.active_target_classes_cv = args.classes.split(',') if args.classes else None

        self.cv_class_filter_options = [None, ["person"], ["car"], ["person", "car"], ["bottle", "cup"]]
        self.cv_current_filter_index = 0
        if self.active_target_classes_cv:
            try: self.cv_current_filter_index = self.cv_class_filter_options.index(self.active_target_classes_cv)
            except ValueError:
                self.cv_class_filter_options.append(self.active_target_classes_cv)
                self.cv_current_filter_index = len(self.cv_class_filter_options) - 1
        else: self.active_target_classes_cv = self.cv_class_filter_options[self.cv_current_filter_index]

        self.video_writer = None
        self.recording_to_file = False
        self.prev_time_calc = 0
        self.fps_smooth_calc = 0.0
        
        self.img_output_dir = os.path.join(args.output_dir, "images")
        self.video_output_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        # Emotion detection settings
        self.emotion_detection_enabled = args.enable_emotion # New arg
        self.emotion_detector_backend = 'opencv' # Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
        self.emotion_frame_interval = 5 # Analyze emotion every N frames to save computation
        self.frame_counter = 0
        self.last_emotion_results = [] # Store last known emotions

        self._print_opencv_instructions()

    def _load_yolo_model(self): # Renamed from _load_model
        try:
            print(f"[INFO] Loading YOLO model '{self.args.model}' on device '{self.args.device}'...")
            model = YOLO(self.args.model)
            model.to(self.args.device)
            # ... (rest of the YOLO model loading and device check logic from previous version) ...
            actual_device_type = "unknown"
            try:
                if self.args.device == 'cpu': actual_device_type = 'cpu'
                elif torch.cuda.is_available() and self.args.device == 'cuda': 
                    dummy_input_tensor = torch.zeros(1, 3, 32, 32, device=self.args.device)
                    with torch.no_grad(): model(dummy_input_tensor) 
                    if hasattr(model, 'model') and next(model.model.parameters(), None) is not None: actual_device_type = next(model.model.parameters()).device.type
                    elif hasattr(model, 'device') and model.device is not None: actual_device_type = model.device.type
                else: 
                    if hasattr(model, 'device') and model.device is not None: actual_device_type = model.device.type
                    else: actual_device_type = self.args.device 
            except Exception as e_dc: print(f"[WARNING] Could not definitively determine model's device via dummy input: {e_dc}") # Shortened error
            if actual_device_type == 'cuda': print(f"[INFO] YOLO Model '{self.args.model}' confirmed on GPU (CUDA).")
            else: print(f"[INFO] YOLO Model '{self.args.model}' loaded. Reported device type: {actual_device_type}.")
            return model
        except Exception as e: print(f"[ERROR] Could not load YOLO model: {e}"); import traceback; traceback.print_exc(); raise SystemExit

    def _init_video_capture(self): # Same
        # ... (implementation from previous response)
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

    def _print_opencv_instructions(self): # Add 'e' for emotion toggle
        print("\n[INFO] OpenCV detection window starting...")
        print("  Press 'q' or ESC in the OpenCV window to quit.")
        print("  Press 's' to save a screenshot.")
        print("  Press 'r' to start/stop recording video to file.")
        print("  Press '+' or '=' to increase confidence threshold.")
        print("  Press '-' or '_' to decrease confidence threshold.")
        print("  Press 'i' to toggle info overlay.")
        print("  Press 'f' to cycle class filters.")
        print("  Press 'e' to toggle emotion detection (if enabled by launch arg).")


    def _analyze_emotions(self, frame_to_analyze):
        """Analyzes emotions in the frame using DeepFace."""
        try:
            # DeepFace.analyze returns a list of dicts, one for each detected face
            # We are interested in 'dominant_emotion' and 'region' (for bounding box)
            # To speed up, use a faster face detector if full analysis is slow
            # enforce_detection=False means if no face is found, it won't raise an error.
            # Use BGR frame directly for DeepFace
            results = DeepFace.analyze(img_path=frame_to_analyze, 
                                       actions=['emotion'], 
                                       detector_backend=self.emotion_detector_backend,
                                       enforce_detection=False,
                                       silent=True) # silent=True to suppress console output from DeepFace
            
            emotion_outputs = []
            if isinstance(results, list): # If faces were found
                for face_data in results:
                    if isinstance(face_data, dict) and 'dominant_emotion' in face_data and 'region' in face_data:
                        region = face_data['region'] # dict with x, y, w, h
                        emotion = face_data['dominant_emotion']
                        # DeepFace emotion distribution might be in face_data['emotion']
                        emotion_confidence = face_data['emotion'].get(emotion) if 'emotion' in face_data else None

                        emotion_outputs.append({
                            "box": (region['x'], region['y'], region['w'], region['h']),
                            "emotion": emotion,
                            "emotion_confidence": emotion_confidence
                        })
            return emotion_outputs
        except Exception as e:
            # This can happen if DeepFace has issues (e.g., model download, unsupported image)
            # Or if no faces are detected and enforce_detection=True (but we set it to False)
            print(f"[WARNING] DeepFace emotion analysis failed: {e}")
            return []


    def _process_frame_for_cv(self, frame):
        # 1. YOLO Object Detection
        yolo_results = self.yolo_model.predict(frame, conf=self.conf_threshold_cv, iou=self.args.iou_thresh, 
                                     classes=self.active_target_classes_cv, verbose=False, device=self.args.device) 
        
        annotated_frame_copy = frame.copy()
        yolo_detections = yolo_results[0].boxes if yolo_results and yolo_results[0].boxes else None
        
        # 2. Emotion Detection (conditionally)
        current_emotion_results_to_draw = []
        if self.emotion_detection_enabled:
            self.frame_counter += 1
            if self.frame_counter % self.emotion_frame_interval == 0 or not self.last_emotion_results:
                # DeepFace expects BGR numpy array
                self.last_emotion_results = self._analyze_emotions(frame) # Pass original frame
            current_emotion_results_to_draw = self.last_emotion_results
        
        # 3. Drawing - now pass emotion results too
        annotated_frame_copy, yolo_detections_count = draw_detections(
            annotated_frame_copy, yolo_detections, self.yolo_model.names, 
            self.args.hide_labels, self.args.hide_conf, self.args.line_thickness,
            emotion_results=current_emotion_results_to_draw # Pass emotion results
        )
        return annotated_frame_copy, yolo_detections_count


    def _handle_cv_keys(self, key, annotated_frame):
        # ... (q, s, r, +, -, i, f keys are the same as before) ...
        if key == ord('q') or key == 27: return False
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S"); filename = os.path.join(self.img_output_dir, f"cv_frame_{timestamp}.jpg")
            cv2.imwrite(filename, annotated_frame); print(f"[INFO] Saved frame to {filename}")
        elif key == ord('r'):
            self.recording_to_file = not self.recording_to_file
            if self.recording_to_file:
                if self.video_writer is None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S"); filename = os.path.join(self.video_output_dir, f"cv_rec_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        
        # New key for toggling emotion detection
        elif key == ord('e'):
            if self.args.enable_emotion: # Only toggle if initially enabled via arg
                self.emotion_detection_enabled = not self.emotion_detection_enabled
                print(f"[INFO] Emotion detection {'ENABLED' if self.emotion_detection_enabled else 'DISABLED'}.")
                if not self.emotion_detection_enabled:
                    self.last_emotion_results = [] # Clear last results when disabling
            else:
                print("[INFO] Emotion detection was not enabled at launch. Cannot toggle with 'e' key.")
        return True

    def run_opencv_window(self): # Same structure as before
        if not self.cap: print("[ERROR] Cannot run, video capture not available."); return
        self.prev_time_calc = time.time()
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
                    annotated_frame, self.fps_smooth_calc, self.args.model, # Use yolo model name from args
                    self.conf_threshold_cv, active_classes_str_cv, 
                    self.recording_to_file, self.cap_w, yolo_detections_count
                )
            cv2.imshow("Object & Emotion Detection (OpenCV)", annotated_frame)
            if self.recording_to_file and self.video_writer: self.video_writer.write(annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_cv_keys(key, annotated_frame): break 
        self.cleanup_opencv_window()

    def cleanup_opencv_window(self): # Same
        print("[INFO] Cleaning up OpenCV window resources...")
        if self.cap: self.cap.release(); self.cap = None
        if self.video_writer: self.video_writer.release(); self.video_writer = None
        cv2.destroyAllWindows()
        print("[INFO] OpenCV window resources cleaned up.")