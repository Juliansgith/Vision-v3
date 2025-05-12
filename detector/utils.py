# detector/utils.py
import cv2
import numpy as np
import os

CLASS_COLORS = {}
EMOTION_COLORS = { # Optional: Define colors for emotions
    "angry": (0, 0, 255),   # Red
    "disgust": (0, 128, 0), # Dark Green
    "fear": (128, 0, 128),# Purple
    "happy": (0, 255, 255), # Yellow
    "sad": (255, 0, 0),     # Blue
    "surprise": (255, 165, 0),# Orange
    "neutral": (128, 128, 128) # Grey
}

def get_color_for_class(class_name):
    if class_name not in CLASS_COLORS:
        hash_val = hash(class_name)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        CLASS_COLORS[class_name] = (b, g, r)
    return CLASS_COLORS[class_name]

def get_color_for_emotion(emotion_name):
    return EMOTION_COLORS.get(emotion_name.lower(), (255, 255, 255)) # Default to white

def draw_detections(frame, yolo_detections, model_names, hide_labels, hide_conf, line_thickness, emotion_results=None):
    """
    Draws YOLO bounding boxes and labels on the frame.
    Optionally, also draws emotion labels if emotion_results are provided.
    emotion_results: list of dicts, where each dict contains 'box' (x,y,w,h for face) and 'emotion'.
    """
    detections_count = {}

    # Draw YOLO object detections
    if yolo_detections:
        for box in yolo_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model_names[class_id]

            detections_count[class_name] = detections_count.get(class_name, 0) + 1
            color = get_color_for_class(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            if not hide_labels:
                label_text = class_name
                if not hide_conf:
                    label_text += f" {confidence:.2f}"
                
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - text_h - baseline - 2), (x1 + text_w, y1 - 2), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw Emotion detections (if any)
    if emotion_results:
        for result in emotion_results:
            x, y, w, h = result['box'] # Face box from DeepFace
            emotion = result['emotion']
            emotion_confidence = result.get('emotion_confidence', None) # If DeepFace provides it

            emotion_color = get_color_for_emotion(emotion)
            # Draw rectangle for face (optional, could be slightly offset from YOLO person box)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, line_thickness)

            # Prepare emotion label
            emotion_label_text = emotion
            if emotion_confidence:
                 emotion_label_text += f" ({emotion_confidence*100:.1f}%)"


            # Position emotion label - typically above the face or near the YOLO person box
            # For simplicity, let's try to put it near the top of the face box from DeepFace
            (text_w, text_h), baseline = cv2.getTextSize(emotion_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Ensure text background is within frame bounds
            text_bg_y1 = max(0, y - text_h - baseline - 2)
            text_bg_x1 = max(0, x)
            text_bg_y2 = max(0, y - 2)
            text_bg_x2 = min(frame.shape[1], x + text_w)

            text_y = max(text_h + baseline, y - baseline) # Ensure text itself is within bounds

            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), emotion_color, -1)
            cv2.putText(frame, emotion_label_text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0) if sum(emotion_color) > 382 else (255,255,255) , 1) # Black/White text

    return frame, detections_count # Return modified frame and YOLO counts


def draw_info_overlay(frame, fps, model_name, conf_thresh, active_classes_str, recording_status, frame_width, counts):
    # ... (same as before)
    info_y_start = 20
    info_y_step = 20
    text_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, info_y_start), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Model: {model_name}", (10, info_y_start + info_y_step), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Conf: {conf_thresh:.2f} (+/-)", (10, info_y_start + 2 * info_y_step), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Filter(f): {active_classes_str}", (10, info_y_start + 3 * info_y_step), font, font_scale, text_color, thickness)

    if recording_status:
        cv2.putText(frame, "REC", (frame_width - 60, info_y_start), font, 0.7, (0, 0, 255), thickness)

    count_y_start = info_y_start + 4 * info_y_step + 10
    # Display YOLO object counts
    yolo_counts_label = "Objects: "
    if not counts:
        yolo_counts_label += "None"
    cv2.putText(frame, yolo_counts_label, (10, count_y_start), font, 0.5, (255,255,0), 1)
    
    current_y_offset = count_y_start + info_y_step
    for i, (cls, count) in enumerate(counts.items()): # counts is yolo_detections_count
         cv2.putText(frame, f"  {cls}: {count}", (10, current_y_offset + i * info_y_step), font, 0.5, (255,255,0), 1)
    return frame

def list_available_cameras(max_cameras_to_check=10): # Same as your last working version
    # ... (implementation from previous response, no changes needed here for emotion)
    available_cameras_info = []
    print("[INFO] Checking for cameras (default backend)...")
    for i in range(max_cameras_to_check):
        try:
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                display_name = f"Webcam {i}"
                print(f"  Found: {display_name}")
                available_cameras_info.append({"name": display_name, "id": i, "backend": "default"})
                cap.release()
            else:
                if available_cameras_info and i > (available_cameras_info[-1]["id"] + 2) :
                    break
        except Exception as e:
            print(f"  Error checking camera index {i} (default): {e}")
            if available_cameras_info and i > (available_cameras_info[-1]["id"] + 2) :
                 break

    if cv2.CAP_DSHOW is not None and os.name == 'nt':
        print("[INFO] Checking for cameras (DSHOW backend on Windows)...")
        default_backend_indices = {cam_info["id"] for cam_info in available_cameras_info}
        for i in range(max_cameras_to_check):
            if i in default_backend_indices and any(cam['id'] == i and cam['backend'] == 'dshow' for cam in available_cameras_info):
                 continue 
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
                if cap is not None and cap.isOpened():
                    display_name = f"Webcam {i} (DSHOW)"
                    print(f"  Found: {display_name}")
                    is_new_entry_by_index = not any(cam['id'] == i for cam in available_cameras_info)
                    if is_new_entry_by_index:
                        available_cameras_info.append({"name": display_name, "id": i, "backend": "dshow"})
                    cap.release()
                else: 
                    if len(available_cameras_info) > 0 and i > (max(c['id'] for c in available_cameras_info) + 2 if available_cameras_info else -1): 
                        break
            except Exception as e:
                print(f"  Error checking camera index {i} (DSHOW): {e}")
                if len(available_cameras_info) > 0 and i > (max(c['id'] for c in available_cameras_info) + 2 if available_cameras_info else -1):
                    break
    
    final_cameras = []
    seen_ids = {} 
    for cam in available_cameras_info:
        if cam['id'] not in seen_ids:
            seen_ids[cam['id']] = cam
            final_cameras.append(cam)
        else: 
            if cam['backend'] == 'dshow' and seen_ids[cam['id']]['backend'] == 'default' and cam['name'] != seen_ids[cam['id']]['name']:
                final_cameras.append(cam) 

    dropdown_choices = [cam["name"] for cam in final_cameras]
    webcams_for_gradio = [(cam["name"], cam["id"]) for cam in final_cameras]

    if not webcams_for_gradio:
        print("[WARNING] No webcams detected by OpenCV.")
    return dropdown_choices, webcams_for_gradio