# gui_launcher.py
import gradio as gr
import os
import sys
import torch
import time
import threading
import logging # Added
from detector.object_detector import ObjectDetector
from detector.utils import list_available_cameras, setup_logging # Added setup_logging
from config import AppSettings, YOLOConfig, TrackerConfig, TrackerParamsSORT, TrackerParamsByteTrack, DeepFaceConfig, DisplayConfig, AppConfig
import subprocess
import tempfile 
import shutil 

DEFAULT_YOLO_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n.engine", "yolov8s.engine", "yolov8m.engine", "yolov8l.engine", "yolov8x.engine"
]
DEEPFACE_BACKENDS = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yunet']
TRACKER_TYPES = ['sort', 'bytetrack']

_initial_dropdown_choices_names, _initial_webcam_id_map = list_available_cameras()
INITIAL_DROPDOWN_CHOICES_SOURCES = _initial_dropdown_choices_names[:]
INITIAL_DROPDOWN_CHOICES_SOURCES.extend(["Use Uploaded Video File", "Use Video Path or URL"])
if not _initial_dropdown_choices_names:
    INITIAL_DROPDOWN_CHOICES_SOURCES = ["Use Uploaded Video File", "Use Video Path or URL"]
WEBCAM_ID_MAP = _initial_webcam_id_map.copy()

detection_thread = None
detector_instance_for_thread = None
downloaded_youtube_video_info = {"path": None, "dir": None} 

TEMP_VIDEO_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_videos")
os.makedirs(TEMP_VIDEO_DOWNLOAD_DIR, exist_ok=True)

# Setup logging at the script level
setup_logging() # Added
logger = logging.getLogger(__name__) # Added


def cleanup_temp_youtube_video():
    global downloaded_youtube_video_info
    if downloaded_youtube_video_info["dir"] and os.path.exists(downloaded_youtube_video_info["dir"]):
        try:
            shutil.rmtree(downloaded_youtube_video_info["dir"])
            logger.info(f"Cleaned up temporary YouTube video directory: {downloaded_youtube_video_info['dir']}")
        except Exception as e:
            logger.warning(f"Could not delete temporary YouTube video directory {downloaded_youtube_video_info['dir']}: {e}", exc_info=True)
    downloaded_youtube_video_info = {"path": None, "dir": None}


def download_youtube_video_to_project_subdir(youtube_url):
    global downloaded_youtube_video_info
    cleanup_temp_youtube_video() 
    unique_subdir_name = f"yt_dl_{int(time.time())}_{os.urandom(4).hex()}"
    current_download_dir = os.path.join(TEMP_VIDEO_DOWNLOAD_DIR, unique_subdir_name)
    os.makedirs(current_download_dir, exist_ok=True)
    
    downloaded_youtube_video_info["dir"] = current_download_dir

    try:
        format_selector = 'bestvideo[ext=mp4][height<=?720][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4][height<=?720][vcodec^=avc]'
        output_template = os.path.join(current_download_dir, "youtube_video.%(ext)s")


        command = [
            'yt-dlp',
            '-f', format_selector,
            '-o', output_template, 
            youtube_url,
            '--no-warnings',
            '--no-playlist',
            '--ignore-config',
            '--progress',
            '--retries', '3',
            '--fragment-retries', '3',
            '--verbose'
        ]
        logger.info(f"Running yt-dlp download command: {' '.join(command)}")
        
        process_result = subprocess.run(command, capture_output=True, text=True, timeout=300)

        logger.debug(f"yt-dlp STDOUT:\n{process_result.stdout}")
        logger.debug(f"yt-dlp STDERR:\n{process_result.stderr}")
        logger.debug(f"yt-dlp return code: {process_result.returncode}")

        if process_result.returncode == 0:
            downloaded_file_path = None
            expected_file_base = os.path.join(current_download_dir, "youtube_video")
            
            possible_extensions = [".mp4", ".mkv", ".webm"] 
            for ext in possible_extensions:
                if os.path.exists(expected_file_base + ext) and os.path.getsize(expected_file_base + ext) > 0:
                    downloaded_file_path = expected_file_base + ext
                    break
            
            if not downloaded_file_path:
                files_in_dir = [f for f in os.listdir(current_download_dir) if os.path.isfile(os.path.join(current_download_dir, f))]
                if files_in_dir:
                    files_in_dir.sort(key=lambda f: os.path.getsize(os.path.join(current_download_dir, f)), reverse=True)
                    largest_file = files_in_dir[0]
                    if os.path.getsize(os.path.join(current_download_dir, largest_file)) > 0:
                         downloaded_file_path = os.path.join(current_download_dir, largest_file)


            if downloaded_file_path:
                logger.info(f"YouTube video downloaded successfully to: {downloaded_file_path}")
                downloaded_youtube_video_info["path"] = downloaded_file_path
                return downloaded_file_path
            else:
                logger.error(f"yt-dlp reported success but could not find a valid downloaded video file in {current_download_dir}.")
                return None
        else:
            logger.error(f"yt-dlp download failed for URL: {youtube_url}. Return code: {process_result.returncode}. STDERR: {process_result.stderr}")
            return None

    except FileNotFoundError:
        logger.error("yt-dlp command not found. Please ensure it's installed and in your system's PATH.", exc_info=True)
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp download command timed out for URL: {youtube_url}.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error downloading YouTube URL {youtube_url} with yt-dlp: {e}", exc_info=True)
        return None
def check_gpu_availability():
    gpu_available = torch.cuda.is_available()
    pytorch_cuda_status_color = "green" if gpu_available else "red"
    pytorch_cuda_symbol = "✅" if gpu_available else "⚠️"
    message = f'**PyTorch CUDA Available:** <font color="{pytorch_cuda_status_color}">{pytorch_cuda_symbol} {"Yes" if gpu_available else "No"}</font>'
    if gpu_available:
        try: message += f" (Device: {torch.cuda.get_device_name(0)})"
        except Exception: message += " (Could not get device name)"
    try:
        import tensorrt
        trt_status_color, trt_symbol, trt_message_part = "green", "✅", "Found"
    except ImportError:
        trt_status_color, trt_symbol, trt_message_part = "orange", "⚠️", "Not Found (Not critical for .engine files)"
    message += f'\n**TensorRT Python Bindings:** <font color="{trt_status_color}">{trt_symbol} {trt_message_part}</font>'
    return message

def get_default_device_option(): return "cuda" if torch.cuda.is_available() else "cpu"


def run_detector_in_thread(current_app_settings: AppSettings):
    global detection_thread, detector_instance_for_thread
    thread_logger = logging.getLogger(__name__ + ".DetectionThread") # Specific logger for the thread
    thread_logger.info("Detection thread started.")
    try:
        detector_instance_for_thread = ObjectDetector(current_app_settings)
        if detector_instance_for_thread.cap and detector_instance_for_thread.cap.isOpened():
            thread_logger.info(">>> OpenCV window should appear now. Press 'q' or 'X' in that window to quit. <<<")
            detector_instance_for_thread.run_opencv_window()
            thread_logger.info("OpenCV detection process finished.")
        else:
            thread_logger.error("Failed to initialize video source in thread.")
    except SystemExit:
        thread_logger.error("Detection process terminated early (SystemExit, e.g. model load error).")
    except Exception as e:
        thread_logger.error(f"Unexpected error in detection thread: {e}", exc_info=True)
    finally:
        thread_logger.info("Detection thread finished execution.")
        if downloaded_youtube_video_info["path"] and \
           current_app_settings.app.source == downloaded_youtube_video_info["path"]:
             cleanup_temp_youtube_video()
        
        detector_instance_for_thread = None
        if threading.current_thread() is detection_thread:
            detection_thread = None

def launch_detection_process(
                            input_source_display, video_file_upload, video_file_path_or_url_text,
                            yolo_model_selection, custom_yolo_model_path,
                            conf_thresh_slider_val, iou_thresh_slider_val, face_person_iou_thresh_slider_val, target_classes_tb_val,
                            hide_conf_cb_val, hide_labels_cb_val, show_track_id_cb_val, line_thickness_slider_val,
                            no_info_overlay_cb_val, device_dd_val,
                            enable_emotion_cb_val, enable_age_gender_cb_val,
                            deepface_backend_dd_val, facial_analysis_interval_slider_val,
                            max_faces_to_analyze_slider_val,
                            async_deepface_cb_val, crop_for_deepface_cb_val,
                            tracker_type_dd_val,
                            sort_max_age_slider_val, sort_min_hits_slider_val, sort_iou_thresh_slider_val,
                            bt_track_thresh_slider_val, bt_track_buffer_slider_val, bt_match_thresh_slider_val,
                            progress=gr.Progress(track_tqdm=True)):
    global detection_thread

    if detection_thread is not None and detection_thread.is_alive():
        return "[STATUS] Detection process is already running. Close OpenCV window first."
    
    cleanup_temp_youtube_video() 
    progress(0, desc="Initializing...") 

    source_value = None

    if input_source_display == "Use Uploaded Video File":
        if video_file_upload is not None:
            source_value = video_file_upload.name
        else:
            err_msg = "[ERROR] 'Use Uploaded Video File' selected, but no file uploaded."
            logger.error(err_msg)
            return err_msg
    elif input_source_display == "Use Video Path or URL":
        if video_file_path_or_url_text:
            path_or_url = video_file_path_or_url_text.strip()
            if "youtube.com/" in path_or_url or "youtu.be/" in path_or_url:
                progress(0.1, desc="Attempting YouTube download...")
                local_yt_file = download_youtube_video_to_project_subdir(path_or_url)
                progress(0.3, desc="YouTube download attempt finished.")
                if local_yt_file:
                    source_value = local_yt_file
                else:
                    # download_youtube_video_to_project_subdir already logs specifics
                    err_msg = f"[ERROR] Could not download YouTube video: {path_or_url}. Check console logs for details."
                    logger.error(f"YouTube download failed for URL: {path_or_url}") # General log here
                    return err_msg
            elif path_or_url.startswith("http://") or path_or_url.startswith("https://"):
                source_value = path_or_url
            elif os.path.exists(path_or_url):
                source_value = path_or_url
            else:
                err_msg = f"[ERROR] Video file path or URL invalid/not found: {path_or_url}"
                logger.error(err_msg)
                return err_msg
        else:
            err_msg = "[ERROR] 'Use Video Path or URL' selected, but no path/URL entered."
            logger.error(err_msg)
            return err_msg
    elif input_source_display in WEBCAM_ID_MAP:
        source_value = str(WEBCAM_ID_MAP[input_source_display])
    else:
        err_msg = f"[ERROR] Invalid input source selected: {input_source_display}."
        logger.error(err_msg)
        return err_msg

    if source_value is None: # Should be caught by earlier checks, but as a safeguard
        err_msg = "[ERROR] Could not determine video source. Please check selection and input."
        logger.error(err_msg)
        return err_msg

    progress(0.4, desc="Loading YOLO model...")
    actual_yolo_model_path_gui = custom_yolo_model_path if custom_yolo_model_path else yolo_model_selection
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    potential_paths = [ actual_yolo_model_path_gui, os.path.join(script_dir, actual_yolo_model_path_gui) ]
    if not os.path.isabs(actual_yolo_model_path_gui):
        potential_paths.append(os.path.join(os.getcwd(), actual_yolo_model_path_gui))

    found_model_path = None
    for p_path in potential_paths:
        if os.path.exists(p_path):
            found_model_path = os.path.abspath(p_path)
            break
    if not found_model_path:
        if downloaded_youtube_video_info["path"]: cleanup_temp_youtube_video()
        err_msg = f"YOLO model file not found: {actual_yolo_model_path_gui}. Checked CWD, script dir, and as provided."
        logger.error(err_msg)
        return f"[ERROR] {err_msg}"
    
    yolo_classes_list = [cls.strip() for cls in target_classes_tb_val.split(',')] if target_classes_tb_val else None

    app_settings = AppSettings(
        app=AppConfig(
            source=source_value,
            output_dir="output" # Default output_dir, can be made configurable in GUI if needed
        ),
        yolo=YOLOConfig(
            model=found_model_path,
            device=device_dd_val,
            conf_thresh=conf_thresh_slider_val,
            iou_thresh=iou_thresh_slider_val,
            classes=yolo_classes_list
        ),
        tracker=TrackerConfig(
            tracker_type=tracker_type_dd_val,
            show_track_id=show_track_id_cb_val,
            sort_params=TrackerParamsSORT(
                sort_max_age=sort_max_age_slider_val,
                sort_min_hits=sort_min_hits_slider_val,
                sort_iou_thresh=sort_iou_thresh_slider_val
            ),
            bytetrack_params=TrackerParamsByteTrack(
                bytetrack_track_thresh=bt_track_thresh_slider_val,
                bytetrack_track_buffer=bt_track_buffer_slider_val,
                bytetrack_match_thresh=bt_match_thresh_slider_val
            )
        ),
        deepface=DeepFaceConfig(
            enable_emotion=enable_emotion_cb_val,
            enable_age_gender=enable_age_gender_cb_val,
            deepface_backend=deepface_backend_dd_val,
            facial_analysis_interval=facial_analysis_interval_slider_val,
            max_faces_to_analyze=max_faces_to_analyze_slider_val,
            face_person_iou_thresh=face_person_iou_thresh_slider_val,
            async_deepface=async_deepface_cb_val,
            crop_for_deepface=crop_for_deepface_cb_val
        ),
        display=DisplayConfig(
            hide_conf=hide_conf_cb_val,
            hide_labels=hide_labels_cb_val,
            line_thickness=line_thickness_slider_val,
            no_info=no_info_overlay_cb_val
        )
    )
    
    logger.info(f"Preparing to launch detection with settings:\n{app_settings.model_dump_json(indent=2)}")
    
    progress(0.5, desc="Starting detection thread...")
    try:
        detection_thread = threading.Thread(target=run_detector_in_thread, args=(app_settings,))
        detection_thread.daemon = True
        detection_thread.start()
        progress(1.0, desc="Detection thread started.")
        logger.info("Detection thread initiated.")
        time.sleep(2.5) 
        
        if detection_thread and detection_thread.is_alive():
            if detector_instance_for_thread and detector_instance_for_thread.cap and detector_instance_for_thread.cap.isOpened():
                logger.info("Detection process started successfully, OpenCV window should be active.")
                return "[STATUS] Detection process started. OpenCV window should appear. Press 'q' in that window to quit."
            else: 
                logger.error("Detection thread started, but failed to initialize video source or critical components. Check console for errors from the thread.")
                detection_thread.join(timeout=3.5) 
                if app_settings.app.source == downloaded_youtube_video_info["path"]:
                     cleanup_temp_youtube_video()
                return "[ERROR] Detection thread started, but failed to initialize video source or critical components. Check console for errors from the thread."
        else: 
            logger.error("Failed to start or sustain detection thread. Check console for errors.")
            if downloaded_youtube_video_info["path"]: cleanup_temp_youtube_video() 
            return "[ERROR] Failed to start or sustain detection thread. Check console for errors."
    except Exception as e:
        logger.error(f"Failed to start detection thread: {e}", exc_info=True)
        if downloaded_youtube_video_info["path"]: cleanup_temp_youtube_video()
        return f"[ERROR] Failed to start detection: {str(e)[:300]}"

# --- (Gradio UI Blocks definition remains the same) ---
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown(
        """
        # 🚀 Real-time Multi-Analysis Detector
        Object Detection (YOLO) + Person Tracking (SORT/ByteTrack) + Facial Analysis (DeepFace). Async options for performance.
        """
    )
    gpu_status_md = gr.Markdown(check_gpu_availability, every=60)

    with gr.Tabs():
        with gr.TabItem("⚙️ Core Settings"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Input Source")
                    default_input_source = INITIAL_DROPDOWN_CHOICES_SOURCES[0]
                    input_source_dd = gr.Dropdown(label="Select Source Type", choices=INITIAL_DROPDOWN_CHOICES_SOURCES, value=default_input_source, interactive=True)
                    
                    video_file_upload_component = gr.File(label="Upload Video File (Used if 'Use Uploaded Video File' selected)", file_types=["video", ".mp4", ".avi", ".mov", ".mkv"])
                    video_file_path_or_url_component = gr.Textbox(label="Video File Path or URL (Used if 'Use Video Path or URL' selected)", placeholder="/path/to/video.mp4 or https://youtube.com/watch?v=...")

                    gr.Markdown("### 🧠 YOLO Model & Device")
                    yolo_model_dd = gr.Dropdown(label="Select YOLO Model/Engine", choices=DEFAULT_YOLO_MODELS, value=DEFAULT_YOLO_MODELS[0], info="Ensure .engine files are present if selected.")
                    custom_yolo_model_tb = gr.Textbox(label="Or Custom YOLO Model Path (.pt/.engine)", placeholder="e.g., /path/custom.pt")
                    device_dd = gr.Radio(label="Device (.pt models)", choices=["cuda", "cpu"], value=get_default_device_option(), info="CUDA for GPU. TRT engines auto-use GPU.")

                with gr.Column(scale=3):
                    gr.Markdown("### 🎯 YOLO Detection Parameters") 
                    with gr.Row():
                        conf_thresh_slider = gr.Slider(label="Initial Confidence", minimum=0.01, maximum=1.0, step=0.01, value=0.4)
                        iou_thresh_slider = gr.Slider(label="IoU Threshold (NMS)", minimum=0.01, maximum=1.0, step=0.01, value=0.5)
                    target_classes_tb = gr.Textbox(label="Initial Target Classes (YOLO, comma-separated)", placeholder="e.g., person,car")
                    
                    gr.Markdown("### 🖥️ Display Options (OpenCV)")
                    with gr.Row():
                        hide_conf_cb = gr.Checkbox(label="Hide YOLO Confidence", value=False)
                        hide_labels_cb = gr.Checkbox(label="Hide All YOLO Labels", value=False)
                        show_track_id_cb = gr.Checkbox(label="Show Person Track IDs Initially", value=True)
                    with gr.Row():
                        line_thickness_slider = gr.Slider(label="Box Line Thickness", minimum=1, maximum=10, step=1, value=2)
                        no_info_overlay_cb = gr.Checkbox(label="Hide Info Overlay by Default", value=False)

        with gr.TabItem("🔬 Facial Analysis & Tracking"): 
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 😊 Facial Analysis (DeepFace)")
                    enable_emotion_cb = gr.Checkbox(label="Enable Emotion Detection", value=True)
                    enable_age_gender_cb = gr.Checkbox(label="Enable Age & Gender Detection", value=True)
                    deepface_backend_dd = gr.Dropdown(label="DeepFace Detector Backend", choices=DEEPFACE_BACKENDS, value='yunet', info="YuNet/OpenCV=faster. RetinaFace/MTCNN=slower, more accurate.")
                    facial_analysis_interval_slider = gr.Slider(label="FA Interval (frames)", minimum=1, maximum=90, step=1, value=20, info="Higher = better FPS, less frequent FA updates.")
                    max_faces_to_analyze_slider = gr.Slider(label="Max Faces for DeepFace (0=All)", minimum=0, maximum=10, step=1, value=2, info="Limits DeepFace processing to N largest faces/crops. 0=no limit.")
                    face_person_iou_thresh_slider = gr.Slider(label="Face-Person Assoc. IoU",minimum=0.1, maximum=0.9, step=0.05, value=0.3, info="Threshold to link DeepFace face to YOLO person.")
                    gr.Markdown("#### FA Performance Options:")
                    async_deepface_cb = gr.Checkbox(label="Enable Asynchronous DeepFace", value=True, info="Runs DeepFace in a separate thread to improve main FPS.")
                    crop_for_deepface_cb = gr.Checkbox(label="Crop Persons for DeepFace (if Async)", value=True, info="Sends person crops (from YOLO) to DeepFace instead of full frame. Effective with Async.")

                with gr.Column(scale=1):
                    gr.Markdown("### 👣 Person Tracking")
                    tracker_type_dd = gr.Dropdown(label="Tracker Type", choices=TRACKER_TYPES, value='sort', interactive=True)
                    with gr.Group(visible=(True)) as sort_args_group: 
                        gr.Markdown("#### SORT Parameters")
                        sort_max_age_slider = gr.Slider(label="SORT Max Age (frames)", minimum=1, maximum=100, step=1, value=30) 
                        sort_min_hits_slider = gr.Slider(label="SORT Min Hits", minimum=1, maximum=20, step=1, value=3)
                        sort_iou_thresh_slider = gr.Slider(label="SORT IoU Threshold", minimum=0.1, maximum=0.9, step=0.05, value=0.3)
                    with gr.Group(visible=(False)) as bytetrack_args_group:
                        gr.Markdown("#### ByteTrack Parameters")
                        bt_track_thresh_slider = gr.Slider(label="Track Thresh (High Conf Det)", minimum=0.1, maximum=0.9, step=0.05, value=0.6)
                        bt_track_buffer_slider = gr.Slider(label="Track Buffer (Frames)", minimum=10, maximum=150, step=5, value=30, info="How long to keep lost tracks.")
                        bt_match_thresh_slider = gr.Slider(label="Match Thresh (Low Conf Det IoU)", minimum=0.1, maximum=0.9, step=0.05, value=0.7)

    start_button = gr.Button("🚀 Start Detection (in OpenCV window)", variant="primary", size="lg")
    output_status_text = gr.Textbox(label="Status", interactive=False, lines=3, max_lines=5)


    def toggle_tracker_args_visibility(tracker_choice):
        return {
            sort_args_group: gr.update(visible=(tracker_choice == 'sort')),
            bytetrack_args_group: gr.update(visible=(tracker_choice == 'bytetrack'))
        }
    tracker_type_dd.change(
        fn=toggle_tracker_args_visibility,
        inputs=tracker_type_dd,
        outputs=[sort_args_group, bytetrack_args_group]
    )

    start_button.click(
        fn=launch_detection_process,
        inputs=[
            input_source_dd, video_file_upload_component, video_file_path_or_url_component,
            yolo_model_dd, custom_yolo_model_tb, # These are component names
            conf_thresh_slider, iou_thresh_slider, face_person_iou_thresh_slider, target_classes_tb, # These are component names
            hide_conf_cb, hide_labels_cb, show_track_id_cb, line_thickness_slider, # Component names
            no_info_overlay_cb, device_dd, # Component names
            enable_emotion_cb, enable_age_gender_cb, # Component names
            deepface_backend_dd, facial_analysis_interval_slider, # Component names
            max_faces_to_analyze_slider, # Component name
            async_deepface_cb, crop_for_deepface_cb, # Component names
            tracker_type_dd, # Component name
            sort_max_age_slider, sort_min_hits_slider, sort_iou_thresh_slider, # Component names
            bt_track_thresh_slider, bt_track_buffer_slider, bt_match_thresh_slider # Component names
        ],
        outputs=output_status_text
    )

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_temp_youtube_video) 
    demo.launch()