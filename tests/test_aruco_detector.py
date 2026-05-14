import numpy as np
from detectors.aruco_detector import ArucoDetector

def test_aruco_detector_init():
    detector = ArucoDetector()
    assert detector is not None

def test_aruco_process_empty_frame():
    detector = ArucoDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    out_frame, result = detector.process_frame(frame)
    
    assert result.should_turn is False
    assert result.has_targets is False
    assert out_frame.shape == (720, 1280, 3)
