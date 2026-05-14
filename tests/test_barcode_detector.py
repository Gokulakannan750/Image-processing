import numpy as np
from detectors.barcode_detector import BarcodeDetector

def test_barcode_detector_init():
    detector = BarcodeDetector()
    assert detector is not None

def test_barcode_process_empty_frame():
    detector = BarcodeDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    out_frame, result = detector.process_frame(frame)
    
    assert result.should_turn is False
    assert result.has_targets is False
    assert out_frame.shape == (480, 640, 3)
