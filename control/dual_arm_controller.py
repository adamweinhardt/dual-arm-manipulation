import time
import json
import threading
from numpy import pi
import numpy as np
import zmq
import argparse
from ur_controller import URController


class DualArmController:
    def __init__(self, left_ip, right_ip):
        """Initialize dual arm controller

        Args:
            left_ip: IP address of left robot
            right_ip: IP address of right robot
            publish_states: Whether to publish robot states via ZMQ
            left_port: ZMQ port for left robot state publishing
            right_port: ZMQ port for right robot state publishing
        """
        # Initialize robots with state publishing if enabled
        self.left = URController(
            left_ip,
        )
        self.right = URController(right_ip)

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
        """Move both arms by the same offset from current positions"""
        left_state = self.left.get_state()
        right_state = self.right.get_state()

        if left_state and right_state:
            pose1 = left_state["pose"] + np.array(offset)
            pose2 = right_state["pose"] + np.array(offset)

            self.left.moveL_ee(pose1)
            self.right.moveL_ee(pose2)

    def move_L_world(self, left_world_pose, right_world_pose):
        """Move both arms to world coordinates simultaneously"""
        self.left.moveL_world(left_world_pose)
        self.right.moveL_world(right_world_pose)

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

    def execute_grasping_from_stream(self, grasping_port=5557, timeout=30.0):
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

    def plot(self):
        """Generate plots for both arms"""
        self.left.plot()
        self.right.plot()
        print("Plots saved for both robots")

    def disconnect(self):
        """Disconnect both arms and cleanup"""
        self.left.disconnect()
        self.right.disconnect()
        print("Both arms disconnected")


def run_test_case():
    """Test case - basic dual arm movements"""
    dual = DualArmController(
        left_ip="192.168.1.33",
        right_ip="192.168.1.66",
    )

    try:
        print("Homing robots...")
        dual.go_home()
        dual.wait_for_all()
        print("Both arms at home")

        # Test offset movement
        print("Testing offset movement...")
        dual.move_L_offset([0.05, 0, -0.05, 0, 0, 0])
        dual.wait_for_all()
        print("Offset movement complete")

        # Keep running for state publishing test
        print("Publishing states for 10 seconds...")
        time.sleep(10)

    finally:
        dual.disconnect()


def run_grasping_case():
    """Grasping case - subscribe to grasping points and execute"""
    dual = DualArmController(
        left_ip="192.168.1.33",
        right_ip="192.168.1.66",
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
