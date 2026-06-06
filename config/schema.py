"""
config/schema.py
================
Pydantic schema for default.yaml.  Validated on every config load.
Unknown top-level keys are allowed (extra='allow') so user extensions don't break.
"""

from __future__ import annotations

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    model_config = {"extra": "allow"}
    index: int = 0
    source: Optional[Union[int, str]] = None
    loop_video: bool = True  # loop when source is a video file
    width: int = Field(1280, ge=64, le=7680)
    height: int = Field(720, ge=64, le=4320)
    fps: int = Field(30, ge=1, le=240)
    timeout_s: float = Field(2.0, gt=0)
    max_reconnect_attempts: int = Field(5, ge=0)
    reconnect_delay_s: float = Field(1.0, ge=0)


class ArucoConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    dictionary: str = "DICT_6X6_250"
    downsample_width: int = Field(1280, ge=64)
    show_rejected: bool = False
    marker_length_m: float = Field(0.20, gt=0)
    turn_trigger_distance_m: float = Field(1.5, gt=0)
    camera_matrix: List[List[float]] = Field(
        default_factory=lambda: [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    dist_coeffs: List[List[float]] = Field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0]]
    )


class BarcodeConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = False
    min_pixel_area: int = Field(20000, ge=0)


class FeatureConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = False
    min_matches: int = Field(15, ge=1)
    match_distance_threshold: float = Field(50, gt=0)


class YoloConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    model: str = "yolov8n.pt"
    confidence: float = Field(0.50, ge=0.0, le=1.0)
    danger_zone_ratio: float = Field(0.12, ge=0.0, le=1.0)
    device: str = "cpu"

    @field_validator("device")
    @classmethod
    def valid_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            raise ValueError(
                f"device must be one of {allowed} or 'cuda:<N>', got '{v}'"
            )
        return v


class PoleConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    width_m: float = Field(0.10, gt=0)
    turn_trigger_distance_m: float = Field(1.5, gt=0)
    hsv_lower1: List[int] = Field(default_factory=lambda: [0, 100, 100])
    hsv_upper1: List[int] = Field(default_factory=lambda: [10, 255, 255])
    hsv_lower2: List[int] = Field(default_factory=lambda: [160, 100, 100])
    hsv_upper2: List[int] = Field(default_factory=lambda: [180, 255, 255])


class RowFollowerConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = False
    exg_threshold: float = Field(0.10, ge=-1.0, le=1.0)
    near_band_top: float = Field(0.55, ge=0.0, le=1.0)
    near_band_bottom: float = Field(0.95, ge=0.0, le=1.0)
    far_band_top: float = Field(0.30, ge=0.0, le=1.0)
    far_band_bottom: float = Field(0.50, ge=0.0, le=1.0)
    rowend_veg_min: float = Field(0.04, ge=0.0, le=1.0)
    rowend_frames: int = Field(5, ge=1)
    path_veg_frac: float = Field(0.35, ge=0.0, le=1.0)


class DetectorsConfig(BaseModel):
    model_config = {"extra": "allow"}
    aruco: ArucoConfig = Field(default_factory=ArucoConfig)
    barcode: BarcodeConfig = Field(default_factory=BarcodeConfig)
    pole: PoleConfig = Field(default_factory=PoleConfig)
    row_follower: RowFollowerConfig = Field(default_factory=RowFollowerConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    yolo: YoloConfig = Field(default_factory=YoloConfig)


class NavigationConfig(BaseModel):
    model_config = {"extra": "allow"}
    turn_cooldown_s: float = Field(10.0, ge=0)
    max_cross_track_error_m: float = Field(0.5, gt=0)
    max_heading_error_deg: float = Field(15.0, gt=0)
    smoothing_alpha: float = Field(0.3, ge=0.0, le=1.0)
    dead_zone_x: int = Field(20, ge=0)
    recovery_timeout_s: float = Field(2.0, ge=0)
    min_state_duration_s: float = Field(0.5, ge=0)


class ControllerConfig(BaseModel):
    model_config = {"extra": "allow"}
    can_interface: str = "virtual"
    can_channel: str = "can0"
    can_bitrate: int = Field(500000, gt=0)
    turn_command_id: int = Field(0x123, ge=0, le=0x7FF)
    turn_command_data: List[int] = Field(
        default_factory=lambda: [1, 0, 0, 0, 0, 0, 0, 0]
    )
    estop_command_id: int = Field(0, ge=0, le=0x7FF)
    estop_command_data: List[int] = Field(
        default_factory=lambda: [255, 255, 0, 0, 0, 0, 0, 0]
    )

    @field_validator("turn_command_data", "estop_command_data")
    @classmethod
    def valid_can_data(cls, v: List[int]) -> List[int]:
        if len(v) != 8:
            raise ValueError("CAN data frames must be exactly 8 bytes")
        for b in v:
            if not (0 <= b <= 255):
                raise ValueError(f"CAN data byte out of range [0,255]: {b}")
        return v


class SafetyConfig(BaseModel):
    model_config = {"extra": "allow"}
    max_stale_frame_ms: float = Field(1500, gt=0)
    enable_auto_recovery: bool = True
    min_fps_warning: float = Field(5.0, ge=0)


class RecordingConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = False
    output_dir: str = "recordings"
    save_raw_frames: bool = True
    save_annotated_frames: bool = True
    save_metadata_json: bool = True


class ValidationConfig(BaseModel):
    model_config = {"extra": "allow"}
    pose_stability_threshold: float = Field(0.8, ge=0.0, le=1.0)
    jitter_warning_m: float = Field(0.05, gt=0)
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)
    enable_stability_overlay: bool = True


class NormalizationConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    method: str = "CLAHE"
    clahe_clip_limit: float = Field(2.0, gt=0)
    clahe_grid_size: List[int] = Field(default_factory=lambda: [8, 8])
    gamma_correction: float = Field(1.2, gt=0)

    @field_validator("method")
    @classmethod
    def valid_method(cls, v: str) -> str:
        allowed = {"CLAHE", "GAMMA", "NONE"}
        if v.upper() not in allowed:
            raise ValueError(f"normalization.method must be one of {allowed}")
        return v.upper()

    @field_validator("clahe_grid_size")
    @classmethod
    def valid_grid(cls, v: List[int]) -> List[int]:
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError(
                "clahe_grid_size must be [rows, cols] with positive values"
            )
        return v


class TestingConfig(BaseModel):
    model_config = {"extra": "allow"}
    active_mode: str = "NORMAL"
    draw_mode_overlay: bool = True


class DashboardConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(5000, ge=1, le=65535)


class DebugConfig(BaseModel):
    model_config = {"extra": "allow"}
    visualize_commands: bool = True
    draw_pose_axes: bool = True
    show_performance: bool = True
    show_latency_graph: bool = True
    draw_path_history: bool = True
    history_maxlen: int = Field(30, ge=1)


class AppConfig(BaseModel):
    """Root configuration schema."""

    model_config = {"extra": "allow"}

    robot_id: str = "robot-0"
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
