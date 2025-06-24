import zmq
import time
import json
import numpy as np
import argparse
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import robotic as ry

from robot_ipc_control.pose_estimation.transform_utils import (
    rotation_matrix_to_quaternion,
)
from robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator
from robot_ipc_control.pose_estimation.scene_utils import make_scene, get_robot_joints
from robot_ipc_control.examples.robot_interface import urtde_to_rai


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer"""

    scene_config_path: str
    robot_configs: List[str] = None
    zmq_ip: str = "127.0.0.1"
    robot_state_port: int = 5556
    update_rate_hz: float = 20.0
    enable_robot_tracking: bool = True
    enable_box_tracking: bool = True
    enable_keyboard_input: bool = True


class ComponentStatus(Enum):
    """Status of visualization components"""

    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 3


class KeyboardHandler(threading.Thread):
    """Thread-safe keyboard input handler"""

    def __init__(self, callback=None):
        super().__init__(name="keyboard-handler", daemon=True)
        self.callback = callback
        self.running = True

    def run(self):
        while self.running:
            try:
                user_input = input()
                if self.callback:
                    self.callback(user_input)
            except (EOFError, KeyboardInterrupt):
                break

    def stop(self):
        self.running = False


class RobotStateTracker:
    """Handles robot state tracking via ZMQ"""

    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = None
        self.status = ComponentStatus.DISCONNECTED
        self.robot_states = {}

    def connect(self):
        """Connect to robot state publisher"""
        try:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.connect(
                f"tcp://{self.config.zmq_ip}:{self.config.robot_state_port}"
            )
            self.status = ComponentStatus.CONNECTED
            print(
                f"✓ Connected to robot state stream on port {self.config.robot_state_port}"
            )
        except Exception as e:
            self.status = ComponentStatus.ERROR
            print(f"✗ Failed to connect to robot state stream: {e}")

    def update(self) -> bool:
        """Update robot states. Returns True if new data received."""
        if self.status != ComponentStatus.CONNECTED or not self.socket:
            return False

        try:
            # Non-blocking receive
            state_data = self.socket.recv_json(flags=zmq.NOBLOCK)

            # Handle single robot or multi-robot data
            if "robot_id" in state_data:
                # Multi-robot format
                robot_id = state_data["robot_id"]
                self.robot_states[robot_id] = state_data
            else:
                # Single robot format - assume robot_0
                self.robot_states["robot_0"] = state_data

            return True

        except zmq.Again:
            # No new data
            return False
        except Exception as e:
            print(f"Robot state update error: {e}")
            return False

    def get_robot_state(self, robot_id: str = "robot_0") -> Optional[Dict]:
        """Get latest state for specified robot"""
        return self.robot_states.get(robot_id)

    def disconnect(self):
        """Clean shutdown"""
        if self.socket:
            self.socket.close()
        self.context.term()


class ComprehensiveVisualizer:
    """Main visualizer class that combines robot and scene tracking"""

    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.running = False

        self.scene_config = self._load_scene_config()

        self.robot_tracker = (
            RobotStateTracker(config) if config.enable_robot_tracking else None
        )
        self.box_estimator = None
        self.keyboard_handler = None

        self.C, self.box_names, self.robot_names = make_scene(self.scene_config)
        self.robot_joint_names = {}

        for robot_name in self.robot_names:
            self.robot_joint_names[robot_name] = get_robot_joints(self.C, robot_name)

        self.component_status = {
            "robot_tracker": ComponentStatus.DISCONNECTED,
            "box_estimator": ComponentStatus.DISCONNECTED,
            "scene": ComponentStatus.DISCONNECTED,
        }

        print(
            f"Scene initialized with {len(self.robot_names)} robots and {len(self.box_names)} boxes"
        )

    def _load_scene_config(self) -> Dict:
        """Load scene configuration from file"""
        try:
            with open(self.config.scene_config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scene config not found: {self.config.scene_config_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in scene config: {e}")

    def _initialize_components(self):
        """Initialize all tracking components"""

        # Initialize robot tracking
        if self.config.enable_robot_tracking and self.robot_tracker:
            self.robot_tracker.connect()
            self.component_status["robot_tracker"] = self.robot_tracker.status

        # Initialize box tracking
        if self.config.enable_box_tracking:
            try:
                port = self.scene_config.get("port", 5555)
                self.box_estimator = BoardPoseEstimator(f"tcp://localhost:{port}")
                self.box_estimator.start()
                self.component_status["box_estimator"] = ComponentStatus.CONNECTED
                print(f"✓ Connected to box pose estimator on port {port}")
            except Exception as e:
                self.component_status["box_estimator"] = ComponentStatus.ERROR
                print(f"✗ Failed to initialize box estimator: {e}")

        # Initialize keyboard handler
        if self.config.enable_keyboard_input:
            self.keyboard_handler = KeyboardHandler(self._handle_keyboard_input)
            self.keyboard_handler.start()

        self.component_status["scene"] = ComponentStatus.CONNECTED

    def _handle_keyboard_input(self, user_input: str):
        """Handle keyboard commands"""
        command = user_input.strip().lower()

        if command == "q" or command == "quit":
            print("Shutdown requested")
            self.running = False
        elif command == "s" or command == "status":
            self._print_status()
        elif command == "r" or command == "reset":
            print("Resetting scene view")
            self.C.view(True)
        elif command == "h" or command == "help":
            self._print_help()
        else:
            print(f"Unknown command: {command}. Type 'h' for help.")

    def _print_help(self):
        """Print available commands"""
        print("\n Available Commands:")
        print("  q, quit  - Shutdown visualizer")
        print("  s, status - Show component status")
        print("  r, reset  - Reset scene view")
        print("  h, help   - Show this help")
        print()

    def _print_status(self):
        """Print current status of all components"""
        print("\n Visualizer Status:")
        for component, status in self.component_status.items():
            status_icon = {
                ComponentStatus.DISCONNECTED: "⚫",
                ComponentStatus.CONNECTING: "🟡",
                ComponentStatus.CONNECTED: "🟢",
                ComponentStatus.ERROR: "🔴",
            }[status]
            print(f"  {component}: {status_icon} {status.name}")

        if self.robot_tracker:
            print(f"  Active robots: {len(self.robot_tracker.robot_states)}")
            for robot_id, state in self.robot_tracker.robot_states.items():
                if robot_id in [f"robot_{i}" for i in range(len(self.robot_configs))]:
                    robot_config = self.robot_configs[int(robot_id.split("_")[1])]
                    print(
                        f"    {robot_id}: {robot_config['robot_ip']}:{robot_config['publisher_port']}"
                    )
                else:
                    print(f"    {robot_id}: connected")

        if self.box_estimator:
            try:
                tracked_boxes = len(self.box_estimator.get_tracked_board_ids())
                print(f"  Tracked boxes: {tracked_boxes}")
            except:
                print(f"  Tracked boxes: unknown")
        print()

    def _update_robot_poses(self):
        """Update robot poses in the scene"""
        if (
            not self.robot_tracker
            or self.component_status["robot_tracker"] != ComponentStatus.CONNECTED
        ):
            return

        if self.robot_tracker.update():
            # Update each robot in the scene
            for i, robot_name in enumerate(self.robot_names):
                robot_id = f"robot_{i}"
                robot_state = self.robot_tracker.get_robot_state(robot_id)

                if robot_state and "Q" in robot_state:
                    joint_positions = urtde_to_rai(robot_state["Q"])
                    joint_names = self.robot_joint_names[robot_name]
                    self.C.setJointState(joint_positions, joint_names)

    def _update_box_poses(self):
        """Update box poses in the scene"""
        if (
            not self.box_estimator
            or self.component_status["box_estimator"] != ComponentStatus.CONNECTED
        ):
            return

        try:
            tracked_ids = self.box_estimator.get_tracked_board_ids()

            for box_id in tracked_ids:
                box_pose = self.box_estimator.get_pose(box_id)
                is_stable = self.box_estimator.is_stable(box_id)

                if int(box_id) < len(self.box_names):
                    box_name = self.box_names[int(box_id)]

                    if is_stable and box_pose is not None:
                        # Update pose
                        position = np.array(box_pose[:3])
                        quaternion = np.array(box_pose[3:])

                        self.C.getFrame(box_name).setRelativePosition(position)
                        self.C.getFrame(box_name).setRelativeQuaternion(quaternion)

                        # Visual feedback for stable tracking
                        self.C.getFrame(box_name).setColor(
                            [0.2, 0.8, 0.2, 1.0]
                        )  # Green for stable
                    else:
                        # Visual feedback for unstable tracking
                        self.C.getFrame(box_name).setColor(
                            [0.8, 0.2, 0.2, 0.3]
                        )  # Red/transparent for unstable

        except Exception as e:
            print(f"Box pose update error: {e}")

    def start(self):
        """Start the visualization loop"""
        print("Starting Comprehensive Robot & Scene Visualizer")
        print("Type 'h' for help, 'q' to quit")

        self._initialize_components()
        self._print_status()

        self.running = True
        update_interval = 1.0 / self.config.update_rate_hz

        try:
            while self.running:
                start_time = time.time()

                # Update all components
                self._update_robot_poses()
                self._update_box_poses()

                # Refresh visualization
                self.C.view(False)

                # Rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n Interrupted by user")
        except Exception as e:
            print(f"\n Unexpected error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components"""
        print(" Shutting down visualizer...")

        self.running = False

        if self.robot_tracker:
            self.robot_tracker.disconnect()

        if self.box_estimator:
            # BoardPoseEstimator might have its own shutdown method
            pass

        if self.keyboard_handler:
            self.keyboard_handler.stop()

        print("✅ Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Robot & Scene Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--zmq-ip",
        type=str,
        default="127.0.0.1",
        help="ZMQ IP address for robot state stream",
    )

    parser.add_argument(
        "--robot-port", type=int, default=5556, help="ZMQ port for robot state stream"
    )

    parser.add_argument(
        "--update-rate",
        type=float,
        default=20.0,
        help="Visualization update rate in Hz",
    )

    parser.add_argument(
        "--disable-robot-tracking",
        action="store_true",
        help="Disable robot pose tracking",
    )

    parser.add_argument(
        "--disable-box-tracking", action="store_true", help="Disable box pose tracking"
    )

    parser.add_argument(
        "--disable-keyboard", action="store_true", help="Disable keyboard input"
    )

    args = parser.parse_args()

    # Hardcoded config path
    config_path = (
        "robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
    )

    # Create configuration
    config = VisualizerConfig(
        scene_config_path=config_path,
        robot_configs=None,
        zmq_ip=args.zmq_ip,
        robot_state_port=args.robot_port,
        update_rate_hz=args.update_rate,
        enable_robot_tracking=not args.disable_robot_tracking,
        enable_box_tracking=not args.disable_box_tracking,
        enable_keyboard_input=not args.disable_keyboard,
    )

    # Create and start visualizer
    visualizer = ComprehensiveVisualizer(config)
    visualizer.start()


if __name__ == "__main__":
    main()
