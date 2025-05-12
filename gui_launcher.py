# gui_launcher.py
import gradio as gr
import argparse
import os
import sys
import torch
import time
import threading # Import threading
from detector.object_detector import ObjectDetector
from detector.utils import list_available_cameras

DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
_initial_dropdown_choices_names, _initial_raw_webcams_map_list = list_available_cameras()

INITIAL_DROPDOWN_CHOICES = _initial_dropdown_choices_names[:]
INITIAL_DROPDOWN_CHOICES.extend(["Upload Video File", "Input Video Path"])

if not _initial_dropdown_choices_names:
    INITIAL_DROPDOWN_CHOICES = ["Upload Video File", "Input Video Path"]

WEBCAM_ID_MAP = {name: cam_id for name, cam_id in _initial_raw_webcams_map_list}
last_video_path = ""

# --- Global variable to manage the detection thread ---
detection_thread = None
# --- Global variable to signal the thread to stop (optional, 'q' key in OpenCV is primary) ---
# stop_detection_flag = threading.Event() # We are relying on 'q' key in OpenCV for now

def check_gpu_availability():
    if torch.cuda.is_available():
        return f"✅ CUDA (GPU) is available! Device: {torch.cuda.get_device_name(0)}"
    else:
        return "⚠️ CUDA (GPU) is NOT available. Model will run on CPU (slower)."

def get_device_option():
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_detector_in_thread(args_namespace):
    """This function will be the target for the new thread."""
    global detection_thread # Allow modification
    print("[THREAD] Detection thread started.")
    try:
        detector = ObjectDetector(args_namespace)
        if detector.cap and detector.cap.isOpened():
            print("[THREAD] >>> OpenCV window should appear now from thread. Press 'q' in that window to quit. <<<")
            detector.run_opencv_window() # This is blocking within this thread
            print("[THREAD] OpenCV detection process finished in thread.")
        else:
            print("[THREAD][ERROR] Failed to initialize video source for ObjectDetector in thread.")
    except SystemExit:
        print("[THREAD][ERROR] Detection process terminated early (SystemExit) in thread.")
    except Exception as e:
        print(f"[THREAD][ERROR] An unexpected error occurred in detection thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[THREAD] Detection thread finished execution.")
        # Clear the global thread variable once done, so we know it's not running
        # This needs careful handling if the Gradio app can be closed while thread is running
        # For now, this simple clear is okay.
        if threading.current_thread() is detection_thread: # ensure only this thread clears it
             detection_thread = None


def launch_detection_process(input_source_display, video_file_upload, video_file_path, model_name,
                           custom_model_path, conf_thresh, iou_thresh, target_classes,
                           hide_conf, hide_labels, line_thickness, no_info_overlay, selected_device,
                           enable_emotion_detection):
    global last_video_path, detection_thread

    if detection_thread is not None and detection_thread.is_alive():
        return "[INFO] Detection process is already running. Please close the OpenCV window first."

    gr.Info("Preparing to launch detection in OpenCV window...")
    # time.sleep(0.1) # Shorter sleep, or remove if gr.Info displays fine
    print("[GUI LAUNCHER] Preparing to launch detection in OpenCV window...")
    
    source_value = None
    # ... (input source logic from previous version - no changes) ...
    if input_source_display == "Upload Video File":
        if video_file_upload is not None: source_value = video_file_upload.name
        else: return "[ERROR] No video file uploaded."
    elif input_source_display == "Input Video Path":
        if video_file_path and os.path.exists(video_file_path):
            source_value = video_file_path
            last_video_path = video_file_path
        else: return f"[ERROR] Video file path invalid: {video_file_path}"
    elif input_source_display in WEBCAM_ID_MAP:
        source_value = str(WEBCAM_ID_MAP[input_source_display])
    else: return f"[ERROR] Invalid input source: {input_source_display}."

    actual_model = custom_model_path if custom_model_path else model_name
    print(f"  YOLO Model: {actual_model}, Device: {selected_device}, Emotions: {'Enabled' if enable_emotion_detection else 'Disabled'}")

    args = argparse.Namespace(
        model=actual_model, source=source_value, conf_thresh=conf_thresh,
        iou_thresh=iou_thresh, classes=target_classes if target_classes else None,
        hide_conf=hide_conf, hide_labels=hide_labels, line_thickness=line_thickness,
        output_dir="output", no_info=no_info_overlay, device=selected_device,
        enable_emotion=enable_emotion_detection
    )

    try:
        print("\n[GUI LAUNCHER] Creating and starting detection thread...")
        # Create and start a new thread for the detection process
        detection_thread = threading.Thread(target=run_detector_in_thread, args=(args,))
        detection_thread.daemon = True # Allow main program to exit even if thread is running (though 'q' should stop it)
        detection_thread.start()
        
        return "[INFO] Detection process started in a separate thread. OpenCV window should appear."

    except Exception as e: # Should not happen here as thread creation is usually robust
        print(f"[ERROR] Failed to start detection thread: {e}")
        import traceback
        traceback.print_exc()
        return f"[ERROR] Failed to start detection: {str(e)[:200]}"

# --- Gradio UI Layout (No changes from previous version) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Real-time Object & Emotion Detector (OpenCV Window)
        Configure settings and click "Start Detection".
        Detection runs in a separate OpenCV window. Press 'q' in that window to quit.
        Emotion detection can be toggled with 'e' key in the OpenCV window if initially enabled.
        """
    )
    gpu_status_md = gr.Markdown(check_gpu_availability())

    with gr.Row():
        with gr.Column(scale=1): 
            gr.Markdown("### Input Source")
            default_input_source = INITIAL_DROPDOWN_CHOICES[0] if INITIAL_DROPDOWN_CHOICES else "Upload Video File"
            input_source_dd = gr.Dropdown(label="Select Input Source", choices=INITIAL_DROPDOWN_CHOICES, value=default_input_source, interactive=True)
            video_file_upload_component = gr.File(label="Upload Video File", file_types=["video"], visible=(default_input_source == "Upload Video File"))
            video_file_path_component = gr.Textbox(label="Enter Video File Path", placeholder="/path/to/your/video.mp4", visible=(default_input_source == "Input Video Path"), value=lambda: last_video_path)

            gr.Markdown("### YOLO Model & Device")
            model_name_dd = gr.Dropdown(label="Select Pre-trained YOLO Model", choices=DEFAULT_MODELS, value=DEFAULT_MODELS[0])
            custom_model_path_tb = gr.Textbox(label="Or Enter Custom YOLO Model Path (.pt)", placeholder="e.g., /path/to/custom_yolo.pt")
            device_dd = gr.Radio(label="Processing Device", choices=["cuda", "cpu"], value=get_device_option(), info="Select CUDA for GPU acceleration.")
            
            gr.Markdown("### Emotion Detection") 
            enable_emotion_cb = gr.Checkbox(label="Enable Emotion Detection (can be intensive)", value=False)

        with gr.Column(scale=1): 
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
    # Removed stop button for now as it's simpler to rely on 'q' in OpenCV window
    output_status_text = gr.Textbox(label="Status", interactive=False, lines=3)

    def update_input_visibility(source_choice):
        upload_visible = source_choice == "Upload Video File"
        path_visible = source_choice == "Input Video Path"
        return {
            video_file_upload_component: gr.update(visible=upload_visible, value=None if not upload_visible else gr.UNCHANGED),
            video_file_path_component: gr.update(visible=path_visible, value="" if not path_visible else gr.UNCHANGED)
        }

    input_source_dd.change(
        fn=update_input_visibility,
        inputs=input_source_dd,
        outputs=[video_file_upload_component, video_file_path_component]
    )
    
    start_button.click(
        fn=launch_detection_process,
        inputs=[
            input_source_dd, video_file_upload_component, video_file_path_component,
            model_name_dd, custom_model_path_tb,
            conf_thresh_slider, iou_thresh_slider, target_classes_tb,
            hide_conf_cb, hide_labels_cb, line_thickness_slider, no_info_overlay_cb,
            device_dd,
            enable_emotion_cb
        ],
        outputs=output_status_text
    )

if __name__ == "__main__":
    demo.launch()