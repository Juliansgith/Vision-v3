import argparse
import os
import torch 
from detector.object_detector import ObjectDetector 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Facial Analysis (CLI)")
    
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file (.pt or .engine)")
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam ID or path to video file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=['cpu', 'cuda'], help="Device for YOLO (.pt models)")
    parser.add_argument("--conf-thresh", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="YOLO IoU threshold (NMS)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated classes to detect (e.g., person,car)")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence scores")
    parser.add_argument("--hide-labels", action="store_true", help="Hide all object labels")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for saves")
    parser.add_argument("--no-info", action="store_true", help="Hide info overlay by default")
    parser.add_argument('--no-track-id', dest='show_track_id', action='store_false', help='Disable person tracking IDs by default.')
    parser.set_defaults(show_track_id=True)

    parser.add_argument("--tracker-type", type=str, default="sort", choices=["sort", "bytetrack"], help="Type of tracker to use for persons.")
    parser.add_argument("--sort-max-age", type=int, default=30, help="SORT: Max frames to keep a track without updates")
    parser.add_argument("--sort-min-hits", type=int, default=3, help="SORT: Min hits to start displaying a track")
    parser.add_argument("--sort-iou-thresh", type=float, default=0.3, help="SORT: IoU threshold for association")
    parser.add_argument("--bytetrack-track-thresh", type=float, default=0.6, help="ByteTrack: High confidence detection threshold.")
    parser.add_argument("--bytetrack-track-buffer", type=int, default=30, help="ByteTrack: Frames to buffer a lost track.") 
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.7, help="ByteTrack: IoU threshold for low score detections.")

    parser.add_argument("--enable-emotion", action="store_true", default=False, help="Enable emotion detection")
    parser.add_argument("--enable-age-gender", action="store_true", default=False, help="Enable age/gender detection")
    parser.add_argument("--deepface-backend", type=str, default="yunet", choices=['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yunet'], help="Face detector backend for DeepFace")
    parser.add_argument("--facial-analysis-interval", type=int, default=20, help="Run facial analysis every N frames")
    parser.add_argument("--max-faces-to-analyze", type=int, default=2, help="Max faces for DeepFace per interval (0=all)")
    parser.add_argument("--face-person-iou-thresh", type=float, default=0.3, help="IoU threshold for face-to-person association")
    parser.add_argument("--async-deepface", action="store_true", default=True, help="Enable asynchronous DeepFace processing.") # Default True
    parser.add_argument('--no-async-deepface', dest='async_deepface', action='store_false', help='Disable asynchronous DeepFace.')
    parser.add_argument("--crop-for-deepface", action="store_true", default=True, help="If async DeepFace, send person crops.") # Default True
    parser.add_argument('--no-crop-for-deepface', dest='crop_for_deepface', action='store_false', help='If async DeepFace, send full frame.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    print("[INFO] Starting detector from Command Line Interface...")
    print(f"[INFO] Arguments: {vars(args)}")
    try:
        detector = ObjectDetector(args)
        detector.run_opencv_window() 
    except SystemExit: print("[INFO CLI] Application terminated early.")
    except Exception as e: print(f"[ERROR CLI] An unexpected error: {e}"); import traceback; traceback.print_exc()
    finally: print("[INFO CLI] Application finished.")