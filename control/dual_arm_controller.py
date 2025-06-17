import time
import numpy as np
from ur_controller import URController


class DualArmController:
    def __init__(self, left_ip, right_ip):
        """Initialize dual arm controller"""
        self.left = URController(left_ip)
        self.right = URController(right_ip)

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
        """Disconnect both arms"""
        self.left.disconnect()
        self.right.disconnect()
        print("Both arms disconnected")


if __name__ == "__main__":
    dual = DualArmController("192.168.1.33", "192.168.1.66")

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

    dual.disconnect()
    dual.plot()
