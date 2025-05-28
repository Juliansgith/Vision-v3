from pydantic import BaseModel, Field
from typing import Optional, List

class YOLOConfig(BaseModel):
    model: str = "yolov8n.pt"
    device: str = "cuda"
    conf_thresh: float = 0.4
    iou_thresh: float = 0.5
    classes: Optional[List[str]] = None

class TrackerParamsSORT(BaseModel):
    sort_max_age: int = 30
    sort_min_hits: int = 3
    sort_iou_thresh: float = 0.3

class TrackerParamsByteTrack(BaseModel):
    bytetrack_track_thresh: float = 0.6
    bytetrack_track_buffer: int = 30
    bytetrack_match_thresh: float = 0.7

class TrackerConfig(BaseModel):
    tracker_type: str = "sort"
    show_track_id: bool = True
    sort_params: TrackerParamsSORT = Field(default_factory=TrackerParamsSORT)
    bytetrack_params: TrackerParamsByteTrack = Field(default_factory=TrackerParamsByteTrack)

class DeepFaceConfig(BaseModel):
    enable_emotion: bool = False
    enable_age_gender: bool = False
    deepface_backend: str = "yunet"
    facial_analysis_interval: int = 20
    max_faces_to_analyze: int = 2
    face_person_iou_thresh: float = 0.3
    async_deepface: bool = True
    crop_for_deepface: bool = True

class DisplayConfig(BaseModel):
    hide_conf: bool = False
    hide_labels: bool = False
    line_thickness: int = 2
    no_info: bool = False # Corresponds to hide_info_overlay

class ReIDConfig(BaseModel):
    reid_model_path: str = "osnet_x0_25_msmt17.pt"
    enable_reid: bool = True
    reid_feature_dim: int = 256
    reid_match_thresh: float = 0.2  # For cosine distance, lower is better
    reid_fuse_weight: float = 0.3  # Weight for ReID cost in fusion

class AppConfig(BaseModel):
    source: str = "0"
    output_dir: str = "output"
    # Add a field for device initialization if needed, e.g. yolo_device_actual: str

class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    deepface: DeepFaceConfig = Field(default_factory=DeepFaceConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
