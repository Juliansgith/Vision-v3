# gui_launcher.py
import gradio as gr
import argparse
import os
import sys
import torch
import time
import threading
from detector.object_detector import ObjectDetector
from detector.utils import list_available_cameras

# Add .engine file options
DEFAULT_YOLO_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n.engine", "yolov8s.engine", "yolov8m.engine", "yolov8l.engine", "yolov8x.engine" # Add engine options
]
# Suggestion: Name your engines more descriptively, e.g., "yolov8n_fp16.engine"
# And ensure these files exist where the script will run or provide full paths.

DEEPFACE_BACKENDS = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe']

_initial_dropdown_choices_names, _initial_raw_webcams_map_list = list_available_cameras()
INITIAL_DROPDOWN_CHOICES = _initial_dropdown_choices_names[:]
INITIAL_DROPDOWN_CHOICES.extend(["Upload Video File", "Input Video Path"])
if not _initial_dropdown_choices_names: INITIAL_DROPDOWN_CHOICES = ["Upload Video File", "Input Video Path"]
WEBCAM_ID_MAP = {name: cam_id for name, cam_id in _initial_raw_webcams_map_list}
last_video_path = ""
detection_thread = None

def check_gpu_availability():
    gpu_available = torch.cuda.is_available()
    message = f"PyTorch CUDA Available: {'✅ Yes' if gpu_available else '⚠️ No'}"
    if gpu_available:
        message += f" (Device: {torch.cuda.get_device_name(0)})"
    # You could add a TensorRT check here if you had a simple way to test it,
    # e.g., try: import tensorrt; message += "\nTensorRT: ✅ Found"
    # except ImportError: message += "\nTensorRT: ⚠️ Not Found (Python bindings)"
    return message

def get_device_option(): return "cuda" if torch.cuda.is_available() else "cpu"

def run_detector_in_thread(args_namespace):
    # ... (same as previous version, no changes needed here for TRT model loading) ...
    global detection_thread
    print("[THREAD] Detection thread started.")
    try:
        detector = ObjectDetector(args_namespace)
        if detector.cap and detector.cap.isOpened():
            print("[THREAD] >>> OpenCV window should appear now from thread. Press 'q' in that window to quit. <<<")
            detector.run_opencv_window()
            print("[THREAD] OpenCV detection process finished in thread.")
        else: print("[THREAD][ERROR] Failed to init video source in thread.")
    except SystemExit: print("[THREAD][ERROR] Detection process terminated early (SystemExit).")
    except Exception as e: print(f"[THREAD][ERROR] Unexpected error in thread: {e}"); import traceback; traceback.print_exc()
    finally:
        print("[THREAD] Detection thread finished execution.")
        if threading.current_thread() is detection_thread: detection_thread = None


def launch_detection_process(input_source_display, video_file_upload, video_file_path, yolo_model_selection, # Renamed from model_name
                           custom_yolo_model_path, conf_thresh, iou_thresh, target_classes, # Renamed
                           hide_conf, hide_labels, line_thickness, no_info_overlay, selected_device,
                           enable_emotion_detection, enable_age_gender_detection, 
                           deepface_backend_selection):
    global last_video_path, detection_thread

    if detection_thread is not None and detection_thread.is_alive():
        return "[INFO] Detection process already running. Close OpenCV window first."

    gr.Info("Preparing to launch detection in OpenCV window...")
    print("[GUI LAUNCHER] Preparing to launch detection...")
    
    source_value = None
    # ... (input source logic - same as before) ...
    if input_source_display == "Upload Video File":
        if video_file_upload is not None: source_value = video_file_upload.name
        else: return "[ERROR] No video file uploaded."
    elif input_source_display == "Input Video Path":
        if video_file_path and os.path.exists(video_file_path):
            source_value = video_file_path; last_video_path = video_file_path
        else: return f"[ERROR] Video file path invalid: {video_file_path}"
    elif input_source_display in WEBCAM_ID_MAP:
        source_value = str(WEBCAM_ID_MAP[input_source_display])
    else: return f"[ERROR] Invalid input source: {input_source_display}."

    # Determine actual YOLO model path (dropdown or custom path)
    actual_yolo_model = custom_yolo_model_path if custom_yolo_model_path else yolo_model_selection
    
    print(f"  YOLO Model: {actual_yolo_model}, Device (for PyTorch models): {selected_device}, "
          f"Emotions: {'Ena' if enable_emotion_detection else 'Dis'}bled, "
          f"Age/Gender: {'Ena' if enable_age_gender_detection else 'Dis'}bled, "
          f"DeepFace Backend: {deepface_backend_selection}")

    args = argparse.Namespace(
        model=actual_yolo_model, # This will be the .pt or .engine file name/path
        source=source_value, conf_thresh=conf_thresh,
        iou_thresh=iou_thresh, classes=target_classes if target_classes else None,
        hide_conf=hide_conf, hide_labels=hide_labels, line_thickness=line_thickness,
        output_dir="output", no_info=no_info_overlay, 
        device=selected_device, # Primarily for .pt model loading, TRT engines often ignore this
        enable_emotion=enable_emotion_detection,
        enable_age_gender=enable_age_gender_detection,
        deepface_backend=deepface_backend_selection
    )

    try:
        print("\n[GUI LAUNCHER] Creating and starting detection thread...")
        detection_thread = threading.Thread(target=run_detector_in_thread, args=(args,))
        detection_thread.daemon = True
        detection_thread.start()
        return "[INFO] Detection process started. OpenCV window should appear."
    except Exception as e:
        print(f"[ERROR] Failed to start detection thread: {e}"); import traceback; traceback.print_exc()
        return f"[ERROR] Failed to start detection: {str(e)[:200]}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown( # ... (same or update with TRT info) ...
        """
        # 🚀 Real-time Multi-Analysis Detector (OpenCV Window)
        Object Detection (YOLO with SORT Tracking, TensorRT option) + Facial Analysis (DeepFace).
        """
    )
    gpu_status_md = gr.Markdown(check_gpu_availability())

    with gr.Row():
        with gr.Column(scale=1): 
            gr.Markdown("### Input Source")
            # ... (Input source UI elements - same as before) ...
            default_input_source = INITIAL_DROPDOWN_CHOICES[0] if INITIAL_DROPDOWN_CHOICES else "Upload Video File"
            input_source_dd = gr.Dropdown(label="Select Input Source", choices=INITIAL_DROPDOWN_CHOICES, value=default_input_source, interactive=True)
            video_file_upload_component = gr.File(label="Upload Video File", file_types=["video"], visible=(default_input_source == "Upload Video File"))
            video_file_path_component = gr.Textbox(label="Enter Video File Path", placeholder="/path/to/your/video.mp4", visible=(default_input_source == "Input Video Path"), value=lambda: last_video_path)


            gr.Markdown("### YOLO Model & Device")
            yolo_model_dd = gr.Dropdown(label="Select Pre-trained YOLO Model/Engine", choices=DEFAULT_YOLO_MODELS, value=DEFAULT_YOLO_MODELS[0]) # Updated choices
            custom_yolo_model_tb = gr.Textbox(label="Or Enter Custom YOLO Model Path (.pt or .engine)", placeholder="e.g., /path/to/custom_yolo.pt") # Updated placeholder
            # Device dropdown is mainly for PyTorch (.pt) models. TensorRT engines are usually GPU-specific.
            device_dd = gr.Radio(label="Processing Device (for .pt models)", choices=["cuda", "cpu"], value=get_device_option(), info="CUDA for GPU. TensorRT engines typically use GPU automatically.")
            
            gr.Markdown("### Facial Analysis (DeepFace)")
            enable_emotion_cb = gr.Checkbox(label="Enable Emotion Detection", value=False)
            enable_age_gender_cb = gr.Checkbox(label="Enable Age & Gender Detection", value=False)
            deepface_backend_dd = gr.Dropdown(label="DeepFace Detector Backend", choices=DEEPFACE_BACKENDS, value='opencv')

        with gr.Column(scale=1): 
            # ... (YOLO Detection Parameters, Display Options - same as before) ...
            gr.Markdown("### YOLO Detection Parameters")
            conf_thresh_slider = gr.Slider(label="Initial Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.4)
            iou_thresh_slider = gr.Slider(label="IoU Threshold (NMS)", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
            target_classes_tb = gr.Textbox(label="Initial Target Classes (YOLO, comma-separated)", placeholder="e.g., person,car")

            gr.Markdown("### Display Options (for OpenCV window)")
            hide_conf_cb = gr.Checkbox(label="Hide YOLO Confidence Scores", value=False)
            hide_labels_cb = gr.Checkbox(label="Hide All YOLO Labels (Boxes Only)", value=False)
            line_thickness_slider = gr.Slider(label="Bounding Box Line Thickness", minimum=1, maximum=10, step=1, value=2)
            no_info_overlay_cb = gr.Checkbox(label="Hide Info Overlay by Default", value=False)

    start_button = gr.Button("🚀 Start Detection (in OpenCV window)", variant="primary")
    output_status_text = gr.Textbox(label="Status", interactive=False, lines=3)

    def update_input_visibility(source_choice): # (same as before)
        # ...
        upload_visible = source_choice == "Upload Video File"; path_visible = source_choice == "Input Video Path"
        return { video_file_upload_component: gr.update(visible=upload_visible, value=None if not upload_visible else gr.UNCHANGED),
                 video_file_path_component: gr.update(visible=path_visible, value="" if not path_visible else gr.UNCHANGED)}

    input_source_dd.change(fn=update_input_visibility, inputs=input_source_dd, outputs=[video_file_upload_component, video_file_path_component])
    
    start_button.click(
        fn=launch_detection_process,
        inputs=[
            input_source_dd, video_file_upload_component, video_file_path_component,
            yolo_model_dd, custom_yolo_model_tb, # Changed from model_name_dd, custom_model_path_tb
            conf_thresh_slider, iou_thresh_slider, target_classes_tb,
            hide_conf_cb, hide_labels_cb, line_thickness_slider, no_info_overlay_cb,
            device_dd,
            enable_emotion_cb, enable_age_gender_cb,
            deepface_backend_dd
        ],
        outputs=output_status_text
    )

if __name__ == "__main__": demo.launch()