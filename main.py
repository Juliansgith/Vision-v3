import argparse
import os
import torch
import logging # Added
from detector.object_detector import ObjectDetector
from config import AppSettings, YOLOConfig, TrackerConfig, TrackerParamsSORT, TrackerParamsByteTrack, DeepFaceConfig, DisplayConfig, AppConfig
from detector.utils import setup_logging # Added
from typing import List # Ensure List is imported for type hinting if not already by other imports

def parse_arguments_to_settings() -> AppSettings:
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Facial Analysis (CLI)")
    
    # YOLO arguments
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file (.pt or .engine)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=['cpu', 'cuda'], help="Device for YOLO (.pt models)")
    parser.add_argument("--conf-thresh", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="YOLO IoU threshold (NMS)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated classes to detect (e.g., person,car)")
    
    # Display arguments
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence scores")
    parser.add_argument("--hide-labels", action="store_true", help="Hide all object labels")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--no-info", action="store_true", help="Hide info overlay by default") # Corresponds to display_config.no_info

    # App arguments
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam ID or path to video file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for saves")

    # Tracker arguments
    parser.add_argument('--no-track-id', dest='show_track_id', action='store_false', help='Disable person tracking IDs by default.')
    parser.set_defaults(show_track_id=True)
    parser.add_argument("--tracker-type", type=str, default="sort", choices=["sort", "bytetrack"], help="Type of tracker to use for persons.")
    parser.add_argument("--sort-max-age", type=int, default=30, help="SORT: Max frames to keep a track without updates")
    parser.add_argument("--sort-min-hits", type=int, default=3, help="SORT: Min hits to start displaying a track")
    parser.add_argument("--sort-iou-thresh", type=float, default=0.3, help="SORT: IoU threshold for association")
    parser.add_argument("--bytetrack-track-thresh", type=float, default=0.6, help="ByteTrack: High confidence detection threshold.")
    parser.add_argument("--bytetrack-track-buffer", type=int, default=30, help="ByteTrack: Frames to buffer a lost track.") 
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.7, help="ByteTrack: IoU threshold for low score detections.")

    # DeepFace arguments
    parser.add_argument("--enable-emotion", action="store_true", default=False, help="Enable emotion detection")
    parser.add_argument("--enable-age-gender", action="store_true", default=False, help="Enable age/gender detection")
    parser.add_argument("--deepface-backend", type=str, default="yunet", choices=['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yunet'], help="Face detector backend for DeepFace")
    parser.add_argument("--facial-analysis-interval", type=int, default=20, help="Run facial analysis every N frames")
    parser.add_argument("--max-faces-to-analyze", type=int, default=2, help="Max faces for DeepFace per interval (0=all)")
    parser.add_argument("--face-person-iou-thresh", type=float, default=0.3, help="IoU threshold for face-to-person association")
    parser.add_argument("--async-deepface", action="store_true", default=True, help="Enable asynchronous DeepFace processing.")
    parser.add_argument('--no-async-deepface', dest='async_deepface', action='store_false', help='Disable asynchronous DeepFace.')
    parser.add_argument("--crop-for-deepface", action="store_true", default=True, help="If async DeepFace, send person crops.")
    parser.add_argument('--no-crop-for-deepface', dest='crop_for_deepface', action='store_false', help='If async DeepFace, send full frame.')

    args = parser.parse_args()

    # Map argparse 'args' to AppSettings
    yolo_classes = [cls.strip() for cls in args.classes.split(',')] if args.classes else None

    app_settings = AppSettings(
        app=AppConfig(
            source=args.source,
            output_dir=args.output_dir
        ),
        yolo=YOLOConfig(
            model=args.model,
            device=args.device, # Already handles cuda availability for default
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            classes=yolo_classes
        ),
        tracker=TrackerConfig(
            tracker_type=args.tracker_type,
            show_track_id=args.show_track_id,
            sort_params=TrackerParamsSORT(
                sort_max_age=args.sort_max_age,
                sort_min_hits=args.sort_min_hits,
                sort_iou_thresh=args.sort_iou_thresh
            ),
            bytetrack_params=TrackerParamsByteTrack(
                bytetrack_track_thresh=args.bytetrack_track_thresh,
                bytetrack_track_buffer=args.bytetrack_track_buffer,
                bytetrack_match_thresh=args.bytetrack_match_thresh
            )
        ),
        deepface=DeepFaceConfig(
            enable_emotion=args.enable_emotion,
            enable_age_gender=args.enable_age_gender,
            deepface_backend=args.deepface_backend,
            facial_analysis_interval=args.facial_analysis_interval,
            max_faces_to_analyze=args.max_faces_to_analyze,
            face_person_iou_thresh=args.face_person_iou_thresh,
            async_deepface=args.async_deepface,
            crop_for_deepface=args.crop_for_deepface
        ),
        display=DisplayConfig(
            hide_conf=args.hide_conf,
            hide_labels=args.hide_labels,
            line_thickness=args.line_thickness,
            no_info=args.no_info
        )
    )
    return app_settings

if __name__ == "__main__":
    setup_logging() # Added
    logger = logging.getLogger(__name__) # Added
    
    app_settings = parse_arguments_to_settings()
    
    # Create output directories
    try:
        os.makedirs(app_settings.app.output_dir, exist_ok=True)
        os.makedirs(os.path.join(app_settings.app.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(app_settings.app.output_dir, "videos"), exist_ok=True)
        logger.info(f"Output directories ensured: {app_settings.app.output_dir}")
    except OSError as e:
        logger.error(f"Error creating output directories: {e}", exc_info=True)
        # Depending on severity, might exit or let ObjectDetector handle it if it also tries to create them.
    
    logger.info("Starting detector from Command Line Interface...")
    logger.info(f"Using configuration: {app_settings.model_dump_json(indent=2)}")
    
    try:
        detector = ObjectDetector(app_settings)
        detector.run_opencv_window() 
    except SystemExit: 
        logger.info("Application terminated early (SystemExit).")
    except Exception as e: 
        logger.error(f"An unexpected error occurred in the main application loop: {e}", exc_info=True)
    finally: 
        logger.info("Application finished.")