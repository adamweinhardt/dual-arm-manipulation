import time
import json
import threading
import numpy as np
import zmq
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

                left_state = self.left.get_state()
                right_state = self.right.get_state()

                if left_state:
                    left_msg = {
                        "timestamp": time.time(),
                        "robot_id": "robot_0",
                        "Q": left_state.get("Q", []),
                        "pos": left_state.get("pos", []),
                        "vel": left_state.get("vel", []),
                    }
                    self.left_publisher.send_json(left_msg)

                if right_state:
                    right_msg = {
                        "timestamp": time.time(),
                        "robot_id": "robot_1",
                        "Q": right_state.get("Q", []),
                        "pos": right_state.get("pos", []),
                        "vel": right_state.get("vel", []),
                    }
                    self.right_publisher.send_json(right_msg)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Publishing error: {e}")
                time.sleep(0.1)

    # === Basic Coordinated Commands ===
    def go_home(self):
        """Move both arms to home position"""
        self.left.go_home()
        self.right.go_home()

    def move_L(self, left_pose, right_pose):
        """Move both arms to different poses simultaneously"""
        self.left.moveL(left_pose)
        self.right.moveL(right_pose)

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

    def disconnect(self):
        """Disconnect both arms and cleanup"""
        if self.publish_states:
            self.stop_publishing()
            self.left_publisher.close()
            self.right_publisher.close()
            self.context.term()

        # Disconnect robots
        self.left.disconnect()
        self.right.disconnect()
        print("Both arms disconnected")


if __name__ == "__main__":
    dual = DualArmController(
        left_ip="192.168.1.33", right_ip="192.168.1.66", publish_states=True
    )

    try:
        # Home both arms
        dual.go_home()
        dual.wait_for_all()
        print("Both arms at home")

        # Move to different positions
        left_pose = [0.3, 0.3, 0.3, 0, 3.14, 0]
        right_pose = [0.3, 0.3, 0.3, 0, 3.14, 0]
        dual.move_L(left_pose, right_pose)
        dual.wait_for_all()
        print("Coordinated movement complete")

        # Keep publishing for a while (for testing)
        print("Publishing states for 10 seconds...")
        time.sleep(10)

    finally:
        dual.disconnect()
        dual.plot()
