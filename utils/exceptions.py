"""
utils/exceptions.py
===================
Centralized custom exceptions for the robotics pipeline.
Provides fine-grained error handling for graceful recovery.
"""

class RoboticsBaseError(Exception):
    """Base exception for all robotics errors."""
    pass

class CameraInitializationError(RoboticsBaseError):
    """Raised when the camera fails to open or is disconnected during init."""
    pass

class CalibrationLoadError(RoboticsBaseError):
    """Raised when camera intrinsic matrices or distortion coefficients fail to load."""
    pass

class DetectorError(RoboticsBaseError):
    """Raised when a computer vision detector encounters an invalid frame or fails catastrophically."""
    pass

class NavigationError(RoboticsBaseError):
    """Raised when the navigation engine receives invalid states or calculations fail."""
    pass

class ControllerError(RoboticsBaseError):
    """Raised when the hardware controller (e.g., CAN bus) fails to initialize or send a command."""
    pass

class SafetyViolationError(RoboticsBaseError):
    """Raised by the Safety Monitor to trigger an immediate Emergency Stop."""
    pass
