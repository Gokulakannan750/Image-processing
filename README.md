# Professional Agricultural Robotics Software

A scalable, production-ready computer vision and autonomous navigation framework for agricultural robots.

## Phase 2 Architecture Overhaul

This project has been heavily refactored for **real-time robustness and determinism**. It introduces signal smoothing, strict state machine transitions, pose filtering, multi-marker support, and robust camera auto-recovery.

### New Features

1. **Navigation Smoothing (`navigation_filter.py`)**: Applies Exponential Moving Average (EMA) and configurable dead zones to raw cross-track pixel errors and steering angles. This eliminates violent steering jitter.
2. **Robust Vehicle State Machine (`vehicle_state.py`)**: Transitions are strictly controlled. The system ensures the robot dwells in a state for a `min_duration_s` to prevent oscillation between DRIVING and TURNING.
3. **Lost Marker Recovery (`recovery_manager.py`)**: If the camera temporarily loses sight of an ArUco marker due to dirt or glare, the robot will enter a `RECOVERING` state and maintain its last known heading. If the timeout is exceeded, the robot transitions to `STOPPED`.
4. **Camera Auto-Recovery (`camera_stream.py`)**: The threaded camera stream detects hardware freezes or timeouts and seamlessly re-initializes `cv2.VideoCapture` under the hood without crashing the application.
5. **Multi-Marker Support**: Detectors now return a `DetectionResult` containing a list of `DetectionTarget` objects. The `DecisionEngine` automatically extracts the highest-priority target for navigation.
6. **Synthetic Simulation (`simulation/`)**: You can test the entire pipeline locally without physical markers or cameras using `python main.py --simulate`.

### Directory Structure

```text
Image_Processing/
├── camera/             # Auto-recovering threaded camera streams
├── config/             # YAML configurations
├── controllers/        # Command priority queue and CAN hardware
├── detectors/          # Multi-marker ArUco/ORB/Barcode detectors
├── navigation/         # Decision engine, state machine, filters, recovery
├── recording/          # Dataset recording & replay
├── safety/             # E-STOP and health monitors
├── simulation/         # Synthetic marker environment
├── tests/              # Pytest suite
├── utils/              # Rotating structured loggers (debug, runtime, navigation, safety)
├── vision/             # Frame pipeline, performance monitors, pose filtering
└── main.py             # Main execution loop
```

## Running the Application

### Live Mode
```bash
python main.py
```

### Synthetic Simulation Mode
```bash
python main.py --simulate
```

### Replay Mode
```bash
python main.py --replay recordings/session_2026.../
```
