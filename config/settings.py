"""
config/settings.py
==================
Central configuration for the Image_Processing system.
Edit the values here rather than hunting through individual modules.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #
    camera_index: int = 0
    """Default OpenCV VideoCapture device index."""

    # ------------------------------------------------------------------ #
    # ArUco detector
    # ------------------------------------------------------------------ #
    aruco_marker_length_m: float = 0.20
    """Physical side length of the printed ArUco marker in metres (20 cm)."""

    aruco_turn_trigger_distance_m: float = 1.5
    """Z-axis distance in metres below which the U-turn is triggered."""

    # Camera intrinsics — replace with real calibration values!
    aruco_camera_matrix: tuple = field(
        default=((800, 0, 640), (0, 800, 360), (0, 0, 1))
    )
    aruco_dist_coeffs: tuple = field(default=((0, 0, 0, 0),))

    # ------------------------------------------------------------------ #
    # Barcode detector
    # ------------------------------------------------------------------ #
    barcode_min_pixel_area: int = 20_000
    """Bounding-box pixel area above which the barcode is considered 'close enough'."""

    # ------------------------------------------------------------------ #
    # Feature (ORB) detector
    # ------------------------------------------------------------------ #
    feature_min_matches: int = 15
    """Minimum number of good ORB matches to confirm a target detection."""

    feature_match_distance_threshold: int = 50
    """Maximum Hamming distance for a match to be considered 'good'."""

    # ------------------------------------------------------------------ #
    # CAN bus / machine controller
    # ------------------------------------------------------------------ #
    can_interface: str = "virtual"
    """python-can bustype, e.g. 'pcan', 'ixxat', 'kvaser', 'virtual'."""

    can_channel: str = "can0"
    """CAN channel string, e.g. 'PCAN_USBBUS1' or 'can0'."""

    can_bitrate: int = 500_000
    """CAN bus speed in bits-per-second."""

    can_turn_command_id: int = 0x123
    """Arbitration ID for the U-turn CAN message."""

    can_turn_command_data: tuple = field(
        default=(0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
    )
    """8-byte CAN payload for the U-turn command."""

    # ------------------------------------------------------------------ #
    # Navigation / row sequencer
    # ------------------------------------------------------------------ #
    turn_cooldown_s: float = 10.0
    """Minimum seconds between consecutive U-turn commands."""


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere:
#   from config.settings import settings
# ---------------------------------------------------------------------------
settings = Settings()
