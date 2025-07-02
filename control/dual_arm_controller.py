import time
import json
import threading
from numpy import pi

import numpy as np
import zmq
import argparse
from ur_controller import URController


class DualArmController:
    def __init__(
        self, left_ip, right_ip, publish_states=False, left_port=5556, right_port=5559
    ):
        """Initialize dual arm controller

        Args:
            left_ip: IP address of left robot
            right_ip: IP address of right robot
            publish_states: Whether to publish robot states via ZMQ
            left_port: ZMQ port for left robot state publishing
            right_port: ZMQ port for right robot state publishing
        """
        self.left = URController(left_ip)
        self.right = URController(right_ip)

        # State publishing setup
        self.publish_states = publish_states
        self.publishing = False
        self.publisher_thread = None

        if self.publish_states:
            self.context = zmq.Context()

            # Create publishers for each robot
            self.left_publisher = self.context.socket(zmq.PUB)
            self.left_publisher.bind(f"tcp://*:{left_port}")

            self.right_publisher = self.context.socket(zmq.PUB)
            self.right_publisher.bind(f"tcp://*:{right_port}")

            print(f" State publishing enabled:")
            print(f"  Left robot:  tcp://{left_ip}:{left_port}")
            print(f"  Right robot: tcp://{right_ip}:{right_port}")

            # Start publishing thread
            self.start_publishing()

    def start_publishing(self):
        """Start the state publishing thread"""
        if not self.publish_states:
            return

        self.publishing = True
        self.publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher_thread.start()
        print("State publishing started")

    def stop_publishing(self):
        """Stop the state publishing thread"""
        if self.publishing:
            self.publishing = False
            if self.publisher_thread:
                self.publisher_thread.join()
            print("State publishing stopped")

    def _publish_loop(self):
        """Main publishing loop - runs in background thread"""
        rate_hz = 20  # 20Hz update rate
        interval = 1.0 / rate_hz

        while self.publishing:
            try:
                start_time = time.time()

                # Get current robot states
                left_state = self.left.get_state()
                right_state = self.right.get_state()

                # Publish left robot state
                if left_state:
                    left_msg = {
                        "timestamp": time.time(),
                        "robot_id": "robot_0",
                        "Q": left_state.get("joints", []),
                        "pos": left_state.get("pose", []),
                        "vel": left_state.get("speed", []),
                    }
                    self.left_publisher.send_json(left_msg)

                # Publish right robot state
                if right_state:
                    right_msg = {
                        "timestamp": time.time(),
                        "robot_id": "robot_1",
                        "Q": right_state.get("joints", []),  # FIX: use 'joints' key
                        "pos": right_state.get("pose", []),  # FIX: use 'pose' key
                        "vel": right_state.get("speed", []),  # FIX: use 'speed' key
                    }
                    self.right_publisher.send_json(right_msg)

                # Rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Publishing error: {e}")
                time.sleep(0.1)  # Brief pause before retry

    # === Basic Coordinated Commands ===
    def go_home(self):
        """Move both arms to home position"""
        self.left.go_home()
        self.right.go_home()

    def move_L(self, left_pose, right_pose):
        """Move both arms to different poses simultaneously"""
        self.left.moveL_ee(left_pose)
        self.right.moveL_ee(right_pose)

    def move_L_offset(self, offset):
        """Move both arms to different poses simultaneously"""
        pose1 = self.left.get_state()["pose"] + np.array(offset)
        pose2 = self.right.get_state()["pose"] + np.array(offset)

        self.left.moveL_ee(pose1)
        self.right.moveL_ee(pose2)

    def move_J(self, left_joints, right_joints):
        """Move both arms to different joint positions simultaneously"""
        self.left.moveJ(left_joints)
        self.right.moveJ(right_joints)

    # === Synchronized Waiting ===
    def wait_for_all(self):
        """Wait until both arms finish all queued commands"""
        self.left.wait_for_commands()
        self.right.wait_for_commands()
        self.left.wait_until_done()
        self.right.wait_until_done()
        time.sleep(0.1)  # Brief pause to ensure states are updated

    def get_states(self):
        """Get states of both arms"""
        return {"left": self.left.get_state(), "right": self.right.get_state()}

    def are_moving(self):
        """Check if either arm is moving"""
        return self.left.is_moving() or self.right.is_moving()

    def plot(self):
        """Generate plots for both arms"""
        self.left.plot()
        self.right.plot()
        # Rename the plots
        import os

        if os.path.exists("robot_plot.png"):
            os.rename("robot_plot.png", "left_arm_plot.png")
        self.right.plot()
        if os.path.exists("robot_plot.png"):
            os.rename("robot_plot.png", "right_arm_plot.png")
        print("Plots saved: left_arm_plot.png, right_arm_plot.png")
        """Subscribe to grasping points and execute coordinated movement"""
        # ZMQ subscriber for grasping points
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect(f"tcp://localhost:{grasping_port}")
        subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        subscriber.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message

        try:
            # Wait for grasping points
            print("Waiting for grasping points...")

            start_time = time.time()
            grasping_data = None

            while time.time() - start_time < timeout:
                try:
                    message = subscriber.recv_json(flags=zmq.NOBLOCK)
                    if "grasping_points" in message and message["grasping_points"]:
                        grasping_data = message["grasping_points"]
                        break
                except zmq.Again:
                    time.sleep(0.1)
                    continue

            if not grasping_data:
                print(f"Timeout: No grasping points received in {timeout}s")
                return False

            # Get the first box's grasping points
            box_id, grasp_info = next(iter(grasping_data.items()))

            print(f"\n=== Executing Grasp for Box {box_id} ===")
            print(f"Pair type: {grasp_info['pair_type']}")
            print(f"Confidence: {grasp_info['confidence']:.2f}")
            print(f"Robot assignment: {grasp_info['robot_assignment']}")

            # Extract points and create poses
            point1 = grasp_info["point1"]  # [x, y, z]
            point2 = grasp_info["point2"]  # [x, y, z]
            robot_assignment = grasp_info["robot_assignment"]

            # Convert to poses (add default orientation)
            default_orientation = [0, 3.14, 0]  # [rx, ry, rz]
            pose1 = point1 + default_orientation
            pose2 = point2 + default_orientation

            # Assign poses to robots (fix string keys)
            if robot_assignment["0"] == "face1":
                left_pose, right_pose = pose1, pose2
                print(f"Left robot → Point1: {point1}")
                print(f"Right robot → Point2: {point2}")
            else:
                left_pose, right_pose = pose2, pose1
                print(f"Left robot → Point2: {point2}")
                print(f"Right robot → Point1: {point1}")

            # Execute coordinated movement
            print("\nExecuting coordinated grasp...")
            self.move_L(left_pose, right_pose)
            self.wait_for_all()
            print("Grasping movement complete!")

            return True

        finally:
            subscriber.close()
            context.term()

    def disconnect(self):
        """Disconnect both arms and cleanup"""
        # Stop publishing first
        if self.publish_states:
            self.stop_publishing()
            self.left_publisher.close()
            self.right_publisher.close()
            self.context.term()

        # Disconnect robots
        self.left.disconnect()
        self.right.disconnect()
        print("Both arms disconnected")


def run_test_case():
    """Test case - basic dual arm movements"""
    dual = DualArmController(
        left_ip="192.168.1.33",
        right_ip="192.168.1.66",
        publish_states=True,
        left_port=5556,
        right_port=5559,
    )

    try:
        dual.go_home()
        dual.wait_for_all()
        # dual.move_L_offset([0.05, 0, -0.3, 0, 0, 0])
        # dual.wait_for_all()

    finally:
        dual.disconnect()


def run_grasping_case():
    """Grasping case - subscribe to grasping points and execute"""
    dual = DualArmController(
        left_ip="192.168.1.33",
        right_ip="192.168.1.66",
        publish_states=True,
        left_port=5556,
        right_port=5559,
    )

    try:
        # Home both arms first
        print("Homing robots...")
        dual.go_home()
        dual.wait_for_all()
        print("Both arms at home")

        # Execute grasping from stream
        success = dual.execute_grasping_from_stream()

        if success:
            # Return home
            print("Returning to home...")
            dual.go_home()
            dual.wait_for_all()
            print("Returned to home position")
        else:
            print("Grasping sequence failed")

    finally:
        dual.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Arm Controller")
    parser.add_argument(
        "--case",
        choices=["test", "grasping"],
        default="test",
        help="Choose the case to run: 'test' for basic movements, 'grasping' for real-time grasping",
    )

    args = parser.parse_args()

    if args.case == "test":
        print("Running test case...")
        run_test_case()
    elif args.case == "grasping":
        print("Running grasping case...")
        run_grasping_case()
