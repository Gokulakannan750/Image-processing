# ai/ — Machine-Learning Detector Modules

This package is reserved for inference-based detectors that complement (or
replace) the classical CV approaches in `detectors/`.

## Suggested additions

| Module | Purpose |
|---|---|
| `ai/yolo_detector.py` | End-of-row detection using YOLOv8/v11 |
| `ai/onnx_detector.py` | Generic ONNX Runtime inference wrapper |
| `ai/depth_estimator.py` | Monocular depth estimation (e.g. MiDaS) |

## Interface contract

Any class placed here must inherit from `detectors.base_detector.BaseDetector`
and implement:

```python
def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, Optional[str]]:
    ...
```

This ensures drop-in compatibility with `main.py`.
