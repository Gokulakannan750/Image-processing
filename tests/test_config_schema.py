"""
tests/test_config_schema.py
Tests for the pydantic AppConfig schema: defaults, validation errors, and edge cases.
"""

import pytest
from pydantic import ValidationError
from config.schema import (
    AppConfig,
    CameraConfig,
    YoloConfig,
    ControllerConfig,
    NormalizationConfig,
    NavigationConfig,
)

# ── Defaults ──────────────────────────────────────────────────────────────────


def test_default_config_is_valid():
    cfg = AppConfig()
    assert cfg.robot_id == "robot-0"
    assert cfg.camera.fps == 30
    assert cfg.detectors.yolo.confidence == 0.50
    assert cfg.controller.estop_command_id == 0


def test_partial_override_fills_defaults():
    cfg = AppConfig(camera={"fps": 60})
    assert cfg.camera.fps == 60
    assert cfg.camera.width == 1280  # default preserved


# ── Camera validation ─────────────────────────────────────────────────────────


def test_camera_fps_out_of_range_raises():
    with pytest.raises(ValidationError):
        CameraConfig(fps=0)  # ge=1


def test_camera_negative_width_raises():
    with pytest.raises(ValidationError):
        CameraConfig(width=0)  # ge=64


# ── YOLO validation ───────────────────────────────────────────────────────────


def test_yolo_confidence_out_of_range():
    with pytest.raises(ValidationError):
        YoloConfig(confidence=1.5)


def test_yolo_invalid_device():
    with pytest.raises(ValidationError):
        YoloConfig(device="tpu")  # not in allowed set


def test_yolo_cuda_device_with_index_allowed():
    cfg = YoloConfig(device="cuda:0")
    assert cfg.device == "cuda:0"


# ── CAN data validation ───────────────────────────────────────────────────────


def test_can_data_wrong_length_raises():
    with pytest.raises(ValidationError):
        ControllerConfig(estop_command_data=[1, 2, 3])  # must be 8 bytes


def test_can_data_byte_out_of_range_raises():
    with pytest.raises(ValidationError):
        ControllerConfig(turn_command_data=[256, 0, 0, 0, 0, 0, 0, 0])


# ── Normalization method ──────────────────────────────────────────────────────


def test_invalid_normalization_method_raises():
    with pytest.raises(ValidationError):
        NormalizationConfig(method="HISTOGRAM")  # not in allowed set


def test_normalization_method_case_insensitive():
    cfg = NormalizationConfig(method="clahe")
    assert cfg.method == "CLAHE"


# ── Navigation ────────────────────────────────────────────────────────────────


def test_smoothing_alpha_out_of_range_raises():
    with pytest.raises(ValidationError):
        NavigationConfig(smoothing_alpha=1.5)


# ── Extra fields allowed ──────────────────────────────────────────────────────


def test_unknown_top_level_key_allowed():
    cfg = AppConfig(**{"custom_field": "hello"})
    assert cfg.model_extra.get("custom_field") == "hello"
