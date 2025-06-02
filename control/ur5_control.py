import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class URController:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.rtde_c = RTDEControlInterface(robot_ip)
        self.rtde_r = RTDEReceiveInterface(robot_ip)
        print(f"Connected to UR robot at {robot_ip}")

    def move_to(self, pose, speed=0.5, acceleration=0.3):
        """Move to [x, y, z, rx, ry, rz] pose"""
        return self.rtde_c.moveL(pose, speed, acceleration)

    def move_joints(self, joints, speed=1.0, acceleration=1.4):
        """Move to joint angles [j0, j1, j2, j3, j4, j5]"""
        return self.rtde_c.moveJ(joints, speed, acceleration)

    def go_home(self):
        """Go to home position"""
        home_joints = [-1.601, -1.727, -2.203, -0.808, 1.595, -0.031]
        print("Going home...")
        return self.move_joints(home_joints)

    def get_pose(self):
        """Get current TCP pose [x, y, z, rx, ry, rz]"""
        return self.rtde_r.getActualTCPPose()

    def get_joints(self):
        """Get current joint angles"""
        return self.rtde_r.getActualQ()

    def get_force(self):
        """Get TCP force/torque"""
        return self.rtde_r.getActualTCPForce()

    def get_speed(self):
        """Get TCP speed"""
        return self.rtde_r.getActualTCPSpeed()

    def stop(self):
        """Stop movement"""
        self.rtde_c.stopL()

    def get_state(self):
        """Get current robot state as dict"""
        return {
            "pose": self.get_pose(),
            "joints": self.get_joints(),
            "force": self.get_force(),
            "speed": self.get_speed(),
        }

    def print_state(self):
        """Print current robot state"""
        state = self.get_state()
        print(f"TCP Pose: [{', '.join([f'{x:.3f}' for x in state['pose']])}]")
        print(f"Joints:   [{', '.join([f'{x:.3f}' for x in state['joints']])}]")
        print(f"Force:    [{', '.join([f'{x:.1f}' for x in state['force']])}]")
        print(f"Speed:    [{', '.join([f'{x:.3f}' for x in state['speed']])}]")

    def disconnect(self):
        """Disconnect from robot"""
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        print("Disconnected from robot")


if __name__ == "__main__":
    robot = URController("172.17.0.2")  # Change IP
    robot.go_home()
    robot.disconnect()
    # target_pose = [0.3, -0.3, 0.3, 0, 3.14, 0]
    # robot.move_to(target_pose, speed=0.2, acceleration=0.1)

    # try:
    #     while True:
    #         robot.print_state()
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     print("\nStopped by user")
    # finally:
    #     robot.disconnect()
