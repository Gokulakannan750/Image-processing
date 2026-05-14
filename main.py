"""
main.py
=======
High-performance robotics entry point.
Integrates threaded camera streams, dynamic detector registry,
decision engine, safety monitoring, and command queue execution.
"""
import argparse
import sys
import time

import cv2

from config.config_manager import config_manager
from utils.logger import setup_logger
from utils.exceptions import RoboticsBaseError

# Setup structured logger before importing components that log during init
log = setup_logger("main")

from camera.camera_stream import CameraStream
from controllers.command_queue import CommandQueue
from controllers.machine_controller import MachineController
from detectors.detector_registry import build_detectors_from_config
from navigation.decision_engine import DecisionEngine
from navigation.command_visualizer import CommandVisualizer
from navigation.vehicle_state import State
from safety.safety_monitor import SafetyMonitor
from vision.frame_pipeline import FramePipeline
from recording.data_recorder import DataRecorder
from recording.replay_system import ReplayCameraStream
from simulation.synthetic_environment import SyntheticEnvironment
from testing.stress_test import StressTestSimulator
from testing.report_generator import ReportGenerator
from dashboard.server import dashboard_state
import dashboard.server as dashboard_server


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professional Agricultural Robotics Stack")
    parser.add_argument("--config",   type=str, default=None,  help="Path to custom YAML config file")
    parser.add_argument("--replay",   type=str, default=None,  help="Path to a recorded session directory")
    parser.add_argument("--simulate", action="store_true",     help="Run in synthetic simulation mode")
    parser.add_argument("--stress",   action="store_true",     help="Run in STRESS TEST mode")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    log.info("=== Initializing Robotics Software Stack ===")

    if args.config:
        config_manager.config_path = args.config
        config_manager.load()

    command_queue = CommandQueue()
    controller = MachineController(command_queue)
    safety_monitor = SafetyMonitor(command_queue)

    registry = build_detectors_from_config()
    vision_pipeline = FramePipeline(registry)

    decision_engine = DecisionEngine(command_queue)
    cmd_visualizer = CommandVisualizer()
    recorder = DataRecorder()
    stress_simulator = StressTestSimulator()
    if args.stress:
        stress_simulator.configure(drop_rate=0.1, lag_ms=50, noise_intensity=10)

    if args.replay:
        log.info(f"STARTING IN REPLAY MODE: {args.replay}")
        fps = config_manager.get("camera.fps", 30)
        camera = ReplayCameraStream(args.replay, fps=fps)
    elif args.simulate:
        log.info("STARTING IN SYNTHETIC SIMULATION MODE")
        fps = config_manager.get("camera.fps", 30)
        camera = SyntheticEnvironment(fps=fps)
    else:
        log.info("STARTING LIVE CAMERA STREAM")
        camera = CameraStream()

    # Start web dashboard
    if config_manager.get("dashboard.enabled", True):
        dashboard_server.start(
            host=config_manager.get("dashboard.host", "0.0.0.0"),
            port=config_manager.get("dashboard.port", 5000),
        )

    log.info("=== System Ready ===")
    log.info("Press 'q' to quit.")

    cv2.namedWindow("Robotics Vision Output", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Robotics Vision Output", cv2.WND_PROP_TOPMOST, 1)

    try:
        with camera:
            while True:
                controller.process_queue()

                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    if args.replay:
                        log.info("Replay finished.")
                        break
                    time.sleep(0.01)
                    safety_monitor.check_health()
                    continue

                if args.stress:
                    frame = stress_simulator.process(frame)
                    if frame is None:
                        log.debug("Stress test: frame dropped.")
                        continue

                safety_monitor.notify_frame_received()
                safety_monitor.check_health()

                # Process Vision
                annotated_frame, result = vision_pipeline.process(frame)
                vision_pipeline.perf_monitor.record_latency(result.frame_latency_ms)

                # Navigation Decisions
                current_state, steering_correction = decision_engine.process_detection(result)

                # Draw Visualizers
                primary = result.primary_target
                cmd_visualizer.draw(
                    annotated_frame,
                    current_state,
                    steering_correction,
                    target_center=(primary.center_x, primary.center_y) if primary else (None, None),
                    obstacles=result.obstacles,
                )

                if controller.is_turning:
                    decision_engine.confirm_turn_complete()
                    controller.is_turning = False

                # Push state to web dashboard
                dashboard_state.update(
                    frame=annotated_frame,
                    vehicle_state=current_state.name,
                    fps=vision_pipeline.perf_monitor.current_fps,
                    latency_ms=result.frame_latency_ms,
                    steering=steering_correction,
                    obstacles=result.obstacles,
                    has_target=result.has_targets,
                    target_distance_m=primary.distance_m if primary else None,
                )

                cv2.imshow("Robotics Vision Output", annotated_frame)

                # Recording
                recorder.record(frame, annotated_frame, result, {
                    "has_targets": result.has_targets,
                    "target_id": primary.id if primary else None,
                    "state": current_state.name,
                    "steering_correction": steering_correction
                })

                if cv2.waitKey(30) & 0xFF == ord("q"):
                    log.info("Quit command received.")
                    break

                # Only reset recovery for marker-loss stops, not obstacle stops
                if current_state == State.STOPPED and not decision_engine._obstacle_blocked:
                    decision_engine.recovery_manager.reset()

    except RoboticsBaseError as e:
        log.critical("ROBOTICS FAULT: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as e:
        log.exception("UNHANDLED EXCEPTION: %s", e)
        sys.exit(1)
    finally:
        log.info("Initiating shutdown sequence...")
        cv2.destroyAllWindows()
        registry.shutdown()
        controller.shutdown()

        if recorder.enabled:
            log.info("Generating session test report...")
            report_path = ReportGenerator.generate_from_session(recorder.session_dir)
            log.info(f"Test Report saved to: {report_path}")

        log.info("=== System Shutdown Complete ===")


if __name__ == "__main__":
    main()
