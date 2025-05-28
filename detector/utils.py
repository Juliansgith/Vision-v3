import cv2
import numpy as np
import os

CLASS_COLORS = {} 
EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (255, 165, 0),
    "neutral": (128, 128, 128)
}
GENDER_COLORS = {
    "man": (230, 216, 173), 
    "woman": (193, 182, 255) 
}

def get_color_for_class(class_name):
    if class_name not in CLASS_COLORS:
        hash_val = hash(class_name + "salt_for_color") 
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        if (r + g + b) < 150:
            r = min(255, r + 70)
            g = min(255, g + 70)
            b = min(255, b + 70)
        CLASS_COLORS[class_name] = (b, g, r)
    return CLASS_COLORS[class_name]

def get_color_for_emotion(emotion_name):
    return EMOTION_COLORS.get(emotion_name.lower(), (200, 200, 200)) 

def get_color_for_gender(gender_name):
    return GENDER_COLORS.get(gender_name.lower(), (200, 200, 200)) 

def get_text_color(bg_color):
    """Choose black or white text based on background color brightness."""
    brightness = sum(bg_color) / 3.0
    return (0,0,0) if brightness > 127 else (255,255,255)


def iou_calc_for_association(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    intersection = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    if union == 0: return 0.0
    return intersection / union


def draw_detections(frame, yolo_detections_boxes, model_names, hide_labels, hide_conf, line_thickness,
                    tracked_persons_data=None, show_person_ids=True):
    yolo_detections_count_dict = {}

    if yolo_detections_boxes is not None:
        for box in yolo_detections_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            class_name = model_names[class_id]

            yolo_detections_count_dict[class_name] = yolo_detections_count_dict.get(class_name, 0) + 1
            
            color = get_color_for_class(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            if not hide_labels or class_name == "person": 
                label_parts = []
                if not hide_labels: 
                     label_parts.append(class_name)
                if not hide_conf and not hide_labels:
                     label_parts.append(f"{confidence:.2f}")
                
                label_text = " ".join(label_parts)

                if label_text: 
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    bg_y1 = max(0, y1 - text_h - baseline - 3) 
                    bg_y2 = y1 - baseline + 2
                    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + 4, bg_y2), color, -1)
                    
                    text_color_on_bg = get_text_color(color)
                    cv2.putText(frame, label_text, (x1 + 2, y1 - baseline - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_on_bg, 1, cv2.LINE_AA)

    if tracked_persons_data and show_person_ids:
        for person_id, data in tracked_persons_data.items():
            px1, py1, px2, py2 = data["box_yolo_sort"]
            
            id_text = f"ID: {person_id}"
            id_color = (255, 0, 255) 
            (id_text_w, id_text_h), id_baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            id_y_pos = py1 - id_baseline - 5 
            
            yolo_label_height_approx = 20 
            if not hide_labels and "person" in yolo_detections_count_dict:
                id_y_pos = py1 - yolo_label_height_approx - id_baseline - 5
            
            if id_y_pos < id_text_h : 
                id_y_pos = py1 + id_text_h + 5
            
            cv2.rectangle(frame, (px1, id_y_pos - id_text_h - 2), (px1 + id_text_w + 4, id_y_pos + id_baseline -2), id_color, -1)
            id_text_color_on_bg = get_text_color(id_color)
            cv2.putText(frame, id_text, (px1 + 2, id_y_pos -2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_text_color_on_bg, 2, cv2.LINE_AA)

            facial_info_texts_for_person = []
            if data["facial_analysis"]:
                face_an = data["facial_analysis"]
                if "emotion" in face_an:
                    emo_text = face_an['emotion']
                    if "emotion_confidence" in face_an and face_an["emotion_confidence"] is not None:
                         emo_text += f" ({face_an['emotion_confidence']*100:.0f}%)"
                    facial_info_texts_for_person.append({"text": emo_text, "color": get_color_for_emotion(face_an['emotion'])})
                
                if "age" in face_an:
                    facial_info_texts_for_person.append({"text": f"Age: {face_an['age']}", "color": (220,220,200)}) 
                
                if "gender" in face_an:
                    gen_text = face_an['gender']
                    if "gender_confidence" in face_an and face_an["gender_confidence"] is not None:
                         gen_text += f" ({face_an['gender_confidence']*100:.0f}%)"
                    facial_info_texts_for_person.append({"text": gen_text, "color": get_color_for_gender(face_an['gender'])})

            text_y_offset_facial = id_y_pos + id_baseline + 5 
            
            for item in facial_info_texts_for_person:
                (text_w, text_h), baseline = cv2.getTextSize(item["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                
                text_x_pos_facial = px1 + 5 

                bg_y1_facial = text_y_offset_facial
                bg_y2_facial = text_y_offset_facial + text_h + baseline + 2
                
                if bg_y2_facial > frame.shape[0] - 5: continue 
                if bg_y2_facial > py2 + 15 and text_y_offset_facial > (py1 + (py2-py1)/2): 
                    break


                cv2.rectangle(frame, (text_x_pos_facial, bg_y1_facial), 
                              (text_x_pos_facial + text_w + 4, bg_y2_facial), 
                              item["color"], -1)
                
                text_color_on_bg_facial = get_text_color(item["color"])
                cv2.putText(frame, item["text"], (text_x_pos_facial + 2, text_y_offset_facial + text_h + 1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color_on_bg_facial, 1, cv2.LINE_AA)
                text_y_offset_facial = bg_y2_facial + 2 

    return frame, yolo_detections_count_dict


def draw_info_overlay(frame, info_elements, recording_status, frame_width, counts_dict):
    info_y_start = 20
    info_y_step = 22 
    text_color_info = (0, 255, 0) 
    font_info = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_info = 0.6
    thickness_info = 1 

    current_y = info_y_start
    for key, value in info_elements.items():
        cv2.putText(frame, f"{key}: {value}", (10, current_y), font_info, font_scale_info, text_color_info, thickness_info, cv2.LINE_AA)
        current_y += info_y_step
    
    if recording_status: 
        cv2.putText(frame, "REC", (frame_width - 70, info_y_start), font_info, 0.8, (0, 0, 255), 2, cv2.LINE_AA) # Bold REC

    current_y += 10 
    cv2.putText(frame, "Objects:", (10, current_y), font_info, font_scale_info - 0.1, (255,255,0), thickness_info, cv2.LINE_AA) # Cyan for counts
    current_y += info_y_step

    if not counts_dict:
        cv2.putText(frame, "  None", (10, current_y), font_info, font_scale_info -0.1, (255,255,0), thickness_info, cv2.LINE_AA)
    else:
        for i, (cls, count) in enumerate(counts_dict.items()):
             cv2.putText(frame, f"  {cls}: {count}", (10, current_y), font_info, font_scale_info -0.1, (255,255,0), thickness_info, cv2.LINE_AA)
             current_y += info_y_step -2 
    return frame

def list_available_cameras(max_cameras_to_check=10):
    available_cameras_info = []
    print("[INFO] Checking for available cameras...")

    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            name = f"Camera {i} (Default Backend, {int(width)}x{int(height)})"
            print(f"  Found: {name}")
            available_cameras_info.append({"name": name, "id": i, "backend_api": "default"})
            cap.release()
        else:
            if i > 0 and all(not c['id'] == j for j in range(i-2, i) for c in available_cameras_info):
                break
    
    if os.name == 'nt':
        backends_to_check_win = {
            "DSHOW": cv2.CAP_DSHOW,
            "MSMF": cv2.CAP_MSMF
        }
        for backend_name, backend_api in backends_to_check_win.items():
            print(f"[INFO] Checking for cameras ({backend_name} backend on Windows)...")
            for i in range(max_cameras_to_check):
                already_exists_better = any(cam['id'] == i and cam['backend_api'] != "default" for cam in available_cameras_info)
                if already_exists_better:
                    continue

                cap = cv2.VideoCapture(i, backend_api)
                if cap is not None and cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    name = f"Camera {i} ({backend_name}, {int(width)}x{int(height)})"
                    
                    existing_default_cam_idx = -1
                    for idx, cam_info_existing in enumerate(available_cameras_info):
                        if cam_info_existing['id'] == i and cam_info_existing['backend_api'] == 'default':
                            existing_default_cam_idx = idx
                            break
                    
                    if existing_default_cam_idx != -1:
                        print(f"  Updating: {name} (overwriting default entry for ID {i})")
                        available_cameras_info[existing_default_cam_idx] = {"name": name, "id": i, "backend_api": backend_api}
                    elif not any(cam['id'] == i for cam in available_cameras_info): # New ID
                        print(f"  Found: {name}")
                        available_cameras_info.append({"name": name, "id": i, "backend_api": backend_api})
                    cap.release()
                else:
                    if i > 0 and all(not c['id'] == j for j in range(i-2, i) for c in available_cameras_info if c['backend_api'] == backend_api):
                        break
    
    def sort_key(cam):
        pref_order = {"DSHOW": 0, "MSMF": 1, "default": 2}
        return (cam["id"], pref_order.get(cam["backend_api"], 99))

    available_cameras_info.sort(key=sort_key)
    
    final_cameras_for_dropdown = []
    seen_ids_for_dropdown = set()
    for cam in available_cameras_info:
        if cam['id'] not in seen_ids_for_dropdown:
            final_cameras_for_dropdown.append(cam)
            seen_ids_for_dropdown.add(cam['id'])

    if not final_cameras_for_dropdown:
        print("[WARNING] No webcams detected by OpenCV.")
        return [], {} 

    dropdown_choices = [cam["name"] for cam in final_cameras_for_dropdown]
    webcam_id_map = {cam["name"]: cam["id"] for cam in final_cameras_for_dropdown}

    print(f"[INFO] Cameras for Gradio: {dropdown_choices}")
    return dropdown_choices, webcam_id_map

import logging

def setup_logging(level=logging.INFO):
    """
    Configures the root logger for the application.
    """
    # Get the root logger.
    logger = logging.getLogger()
    
    # Set the logging level.
    logger.setLevel(level)
    
    # Prevent multiple handlers if setup_logging is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create a console handler (StreamHandler).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level) # Set level for the handler as well
    
    # Define a log format.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger.
    logger.addHandler(console_handler)
    
    # You can also add a FileHandler here if needed in the future
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG) # Example: log more details to file
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    logger.info("Logging setup complete.") # Initial log to confirm setup