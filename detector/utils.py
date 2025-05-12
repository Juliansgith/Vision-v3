# detector/utils.py
import cv2
import numpy as np
import os

CLASS_COLORS = {} # (same as before)
EMOTION_COLORS = { # (same as before)
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (255, 165, 0),
    "neutral": (128, 128, 128)
}
GENDER_COLORS = { # Optional
    "man": (255, 182, 193), # Light Pink for Man (example)
    "woman": (173, 216, 230) # Light Blue for Woman (example)
}


def get_color_for_class(class_name): # (same as before)
    if class_name not in CLASS_COLORS:
        hash_val = hash(class_name); r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8; b = hash_val & 0x0000FF
        CLASS_COLORS[class_name] = (b, g, r)
    return CLASS_COLORS[class_name]

def get_color_for_emotion(emotion_name): # (same as before)
    return EMOTION_COLORS.get(emotion_name.lower(), (255, 255, 255))

def get_color_for_gender(gender_name): # New
    return GENDER_COLORS.get(gender_name.lower(), (200, 200, 200))


def draw_detections(frame, yolo_detections, model_names, hide_labels, hide_conf, line_thickness, 
                    facial_analysis_results=None, tracked_persons_info=None):
    yolo_detections_count = {}

    # Draw YOLO object detections
    if yolo_detections:
        for box in yolo_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model_names[class_id]

            yolo_detections_count[class_name] = yolo_detections_count.get(class_name, 0) + 1
            color = get_color_for_class(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            if not hide_labels:
                label_text = class_name
                if not hide_conf:
                    label_text += f" {confidence:.2f}"
                
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w + 2, y1 -2), color, -1) # Adjusted for padding
                cv2.putText(frame, label_text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw Facial Analysis results (Emotion, Age, Gender)
    # And associate with tracked persons if possible
    if facial_analysis_results:
        for face_info in facial_analysis_results:
            fx, fy, fw, fh = face_info['box'] # Face box from DeepFace
            
            display_texts = []
            if "emotion" in face_info:
                emotion_text = face_info['emotion']
                if "emotion_confidence" in face_info and face_info["emotion_confidence"] is not None:
                    emotion_text += f" ({face_info['emotion_confidence']*100:.0f}%)"
                display_texts.append({"text": emotion_text, "color": get_color_for_emotion(face_info['emotion'])})
            
            if "age" in face_info:
                display_texts.append({"text": f"Age: {face_info['age']}", "color": (200,200,200)}) # Grey for age
            
            if "gender" in face_info:
                gender_text = face_info['gender']
                if "gender_confidence" in face_info and face_info["gender_confidence"] is not None:
                     gender_text += f" ({face_info['gender_confidence']*100:.0f}%)"
                display_texts.append({"text": gender_text, "color": get_color_for_gender(face_info['gender'])})

            # Drawing facial analysis text
            text_y_offset = fy # Start drawing above the DeepFace detected face box
            for item in display_texts:
                (text_w, text_h), baseline = cv2.getTextSize(item["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                bg_y1 = max(0, text_y_offset - text_h - baseline - 2)
                bg_x1 = max(0, fx)
                bg_y2 = max(0, text_y_offset - 2)
                bg_x2 = min(frame.shape[1] -1 , fx + text_w + 2)
                
                text_draw_y = max(text_h + baseline, text_y_offset - baseline -2)

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), item["color"], -1)
                text_color = (0,0,0) if sum(item["color"]) > 382 else (255,255,255) # Contrast
                cv2.putText(frame, item["text"], (fx + 2, text_draw_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                text_y_offset = bg_y1 # Move next text item above the current one's background

    # Draw Tracked Person IDs
    if tracked_persons_info:
        for person_info in tracked_persons_info:
            person_id = person_info["id"]
            # centroid = person_info["centroid"] # Can use this if no box
            # For now, let's draw ID near the top-left of the person's YOLO box
            if "box" in person_info and person_info["box"]:
                px1, py1, _, _ = person_info["box"]
                id_text = f"ID: {person_id}"
                (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Position above YOLO label or at top-left of box if no space above
                yolo_label_height_approx = 20 # Approx height of YOLO label + padding
                id_y_pos = py1 - yolo_label_height_approx - 5 
                if id_y_pos < text_h : # If too close to top of frame, put inside box
                    id_y_pos = py1 + text_h + 5
                
                cv2.putText(frame, id_text, (px1 + 5, id_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # Magenta for ID

    return frame, yolo_detections_count


def draw_info_overlay(frame, fps, model_name, conf_thresh, active_classes_str, recording_status, frame_width, counts):
    # ... (same as before)
    info_y_start = 20; info_y_step = 20; text_color = (0, 255, 0); font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 2
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, info_y_start), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Model: {model_name}", (10, info_y_start + info_y_step), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Conf: {conf_thresh:.2f} (+/-)", (10, info_y_start + 2 * info_y_step), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Filter(f): {active_classes_str}", (10, info_y_start + 3 * info_y_step), font, font_scale, text_color, thickness)
    if recording_status: cv2.putText(frame, "REC", (frame_width - 60, info_y_start), font, 0.7, (0, 0, 255), thickness)
    count_y_start = info_y_start + 4 * info_y_step + 10
    yolo_counts_label = "Objects: "; current_y_offset = count_y_start + info_y_step
    if not counts: yolo_counts_label += "None"
    cv2.putText(frame, yolo_counts_label, (10, count_y_start), font, 0.5, (255,255,0), 1)
    for i, (cls, count) in enumerate(counts.items()):
         cv2.putText(frame, f"  {cls}: {count}", (10, current_y_offset + i * info_y_step), font, 0.5, (255,255,0), 1)
    return frame

def list_available_cameras(max_cameras_to_check=10): # (same as before)
    # ... (implementation from previous response)
    available_cameras_info = []
    print("[INFO] Checking for cameras (default backend)...")
    for i in range(max_cameras_to_check):
        try:
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                display_name = f"Webcam {i}"; print(f"  Found: {display_name}")
                available_cameras_info.append({"name": display_name, "id": i, "backend": "default"}); cap.release()
            else:
                if available_cameras_info and i > (available_cameras_info[-1]["id"] + 2) : break
        except Exception as e:
            print(f"  Error checking camera index {i} (default): {e}")
            if available_cameras_info and i > (available_cameras_info[-1]["id"] + 2) : break
    if cv2.CAP_DSHOW is not None and os.name == 'nt':
        print("[INFO] Checking for cameras (DSHOW backend on Windows)...")
        default_backend_indices = {cam_info["id"] for cam_info in available_cameras_info}
        for i in range(max_cameras_to_check):
            if i in default_backend_indices and any(cam['id'] == i and cam['backend'] == 'dshow' for cam in available_cameras_info): continue 
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
                if cap is not None and cap.isOpened():
                    display_name = f"Webcam {i} (DSHOW)"; print(f"  Found: {display_name}")
                    is_new_entry_by_index = not any(cam['id'] == i for cam in available_cameras_info)
                    if is_new_entry_by_index: available_cameras_info.append({"name": display_name, "id": i, "backend": "dshow"})
                    cap.release()
                else: 
                    if len(available_cameras_info) > 0 and i > (max(c['id'] for c in available_cameras_info) + 2 if available_cameras_info else -1): break
            except Exception as e:
                print(f"  Error checking camera index {i} (DSHOW): {e}")
                if len(available_cameras_info) > 0 and i > (max(c['id'] for c in available_cameras_info) + 2 if available_cameras_info else -1): break
    final_cameras = []; seen_ids = {} 
    for cam in available_cameras_info:
        if cam['id'] not in seen_ids: seen_ids[cam['id']] = cam; final_cameras.append(cam)
        else: 
            if cam['backend'] == 'dshow' and seen_ids[cam['id']]['backend'] == 'default' and cam['name'] != seen_ids[cam['id']]['name']: final_cameras.append(cam) 
    dropdown_choices = [cam["name"] for cam in final_cameras]
    webcams_for_gradio = [(cam["name"], cam["id"]) for cam in final_cameras]
    if not webcams_for_gradio: print("[WARNING] No webcams detected by OpenCV.")
    return dropdown_choices, webcams_for_gradio