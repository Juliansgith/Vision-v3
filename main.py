import argparse
from detector.object_detector import ObjectDetector 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time Object Detection with YOLO and OpenCV")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to YOLO model file or model name (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: webcam ID (e.g., 0) or path to video file")
    parser.add_argument("--conf-thresh", type=float, default=0.4,
                        help="Confidence threshold for detections (0.0 to 1.0)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for Non-Maximum Suppression")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated list of class names to detect (e.g., person,car). Default: all classes via model.")
    parser.add_argument("--hide-conf", action="store_true",
                        help="Hide confidence scores in object labels")
    parser.add_argument("--hide-labels", action="store_true",
                        help="Hide object labels entirely (shows only bounding boxes)")
    parser.add_argument("--line-thickness", type=int, default=2,
                        help="Thickness of bounding box lines")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output images and videos")
    parser.add_argument("--no-info", action="store_true",
                        help="Hide the on-screen informational display by default")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    try:
        detector = ObjectDetector(args)
        detector.run()
    except SystemExit:
        print("[INFO] Application terminated early due to an error during initialization.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")