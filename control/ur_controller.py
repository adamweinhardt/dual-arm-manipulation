import time
import threading
import queue
from numpy import pi
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robot_ipc_control.pose_estimation.transform_utils import (
    rvec_to_rotmat,
    rotmat_to_rvec,
    quat_to_rotmat,
)


class URController(threading.Thread):
    def __init__(self, ip):
        super().__init__(daemon=True)
        self.ip = ip

        if self.ip == "192.168.1.66":
            self.robot_id = 1
            self.robot_config = np.load(
                "/home/weini/code/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_right.npy"
            )
        elif self.ip == "192.168.1.33":
            self.robot_id = 0
            self.robot_config = np.load(
                "/home/weini/code/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_left.npy"
            )
        else:
            self.robot_id = None
            self.robot_config = None

        # Command queue for threading
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()

        # Robot connections
        self.rtde_control = RTDEControlInterface(self.ip)
        self.rtde_receive = RTDEReceiveInterface(self.ip)

        # Defaults
        self.home_joints = [
            -pi / 2.0,
            -pi / 2.0,
            pi / 2.0,
            -pi / 2.0,
            -pi / 2.0,
            pi,
        ]
        self.ee2marker = np.array(
            [-0.0064, 0.05753, -0.1149, -0.69923, -0.0101, -0.00407, 0.71481]
        )
        self.ee2marker_offset = np.array([0, 0.05753, -0.10, 0, 0, 0])
        self.default_speed = 1.0
        self.default_acceleration = 0.5
        self.default_joint_speed = 1.0
        self.default_joint_acceleration = 1.4

        # Data recording
        self.data = []

        # Start the control thread
        self.start()

    def run(self):
        """Main control thread loop - processes movement commands"""
        while not self._stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)

                recording_thread = threading.Thread(
                    target=self._record_during_movement, daemon=True
                )
                recording_thread.start()

                command()

                self.command_queue.task_done()
            except queue.Empty:
                continue

    def _record_during_movement(self):
        """Records robot state data while movement is happening"""
        time.sleep(0.1)

        for _ in range(100):
            state = self.get_state()
            self.data.append(state)
            time.sleep(0.05)

            if not self.is_moving():
                break

    def _queue_command(self, command):
        """Add a movement command to the execution queue"""
        self.command_queue.put(command)

    def moveL_ee(self, pose):
        command = lambda: self.rtde_control.moveL(
            pose, self.default_speed, self.default_acceleration
        )
        self._queue_command(command)

    def world_2_robot_pose(self, world_pose_6d, robot_base_transform):
        """Convert world pose to robot coordinate frame"""
        # Ensure we have a numpy array
        world_pose_6d = np.array(world_pose_6d)

        world_T = np.eye(4)
        world_T[:3, 3] = world_pose_6d[:3]
        world_T[:3, :3] = rvec_to_rotmat(world_pose_6d[3:])

        robot_base_inv = np.linalg.inv(robot_base_transform)
        robot_T = robot_base_inv @ world_T

        robot_pos = robot_T[:3, 3]
        robot_rvec = rotmat_to_rvec(robot_T[:3, :3])

        return np.array(
            [
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_rvec[0],
                robot_rvec[1],
                robot_rvec[2],
            ]
        )

    def moveL_world(self, world_pose_6d):
        if self.robot_config is None:
            print(f"ERROR: No robot configuration loaded for IP {self.ip}")
            return False

        if isinstance(world_pose_6d, np.ndarray):
            world_pose_6d = world_pose_6d.tolist()

        if len(world_pose_6d) != 6:
            print(
                f"ERROR: world_pose_6d must have 6 elements, got {len(world_pose_6d)}"
            )
            return False

        try:
            robot_pose_6d = self.world_2_robot_pose(world_pose_6d, self.robot_config)

            self.moveL_ee(robot_pose_6d)

        except Exception as e:
            print(f"ERROR in moveL_world: {e}")

    def moveL_gripper_world(self, gripper_world_pose_6d):
        if self.robot_config is None:
            print(f"ERROR: No robot configuration loaded for IP {self.ip}")
            return False

        if not hasattr(self, "ee2marker"):
            print(f"ERROR: ee2marker calibration not loaded for robot {self.robot_id}")
            return False

        # Ensure we have a numpy array
        gripper_world_pose_6d = np.array(gripper_world_pose_6d)

        if len(gripper_world_pose_6d) != 6:
            print(
                f"ERROR: gripper_world_pose_6d must have 6 elements, got {len(gripper_world_pose_6d)}"
            )
            return False

        try:
            self.moveL_world(gripper_world_pose_6d + self.ee2marker_offset)

        except Exception as e:
            print(f"ERROR in moveL_gripper_world: {e}")

    def speedL(self, speed_vector, acceleration=0.5, time_duration=0.1):
        command = lambda: self.rtde_control.speedL(
            speed_vector, acceleration, time_duration
        )
        self._queue_command(command)

    def speedStop(self):
        command = lambda: self.rtde_control.speedStop()
        self._queue_command(command)

    def moveJ(self, joints):
        command = lambda: self.rtde_control.moveJ(
            joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self._queue_command(command)

    def go_home(self):
        command = lambda: self.rtde_control.moveJ(
            self.home_joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self._queue_command(command)

    def get_state(self):
        pose = self.rtde_receive.getActualTCPPose()
        pose_world = self.world_2_robot_pose(pose, self.robot_config)
        return {
            "pose": pose,
            "pose_world": pose_world,
            "joints": self.rtde_receive.getActualQ(),
            "speed": self.rtde_receive.getActualTCPSpeed(),
            "force": self.rtde_receive.getActualTCPForce(),
        }

    def is_moving(self):
        speeds = self.rtde_receive.getActualTCPSpeed()
        return np.linalg.norm(speeds, 2) > 0.01

    def wait_for_commands(self):
        self.command_queue.join()

    def wait_until_done(self):
        time.sleep(0.2)

        if not self.is_moving():
            state = self.get_state()
            self.data.append(state)

    def disconnect(self):
        self._stop_event.set()
        self.join(timeout=2)
        self.rtde_control.disconnect()
        self.rtde_receive.disconnect()

    def plot(self):
        if not self.data:
            print("No data to plot")
            return

        poses = np.array([state["pose"] for state in self.data])

        plt.figure(figsize=(12, 8))

        # TCP position
        plt.subplot(2, 1, 1)
        plt.plot(poses[:, 0], label="X", linewidth=2)
        plt.plot(poses[:, 1], label="Y", linewidth=2)
        plt.plot(poses[:, 2], label="Z", linewidth=2)
        plt.title("TCP Position")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # TCP rotation
        plt.subplot(2, 1, 2)
        plt.plot(poses[:, 3], label="RX", linewidth=2)
        plt.plot(poses[:, 4], label="RY", linewidth=2)
        plt.plot(poses[:, 5], label="RZ", linewidth=2)
        plt.title("TCP Orientation")
        plt.ylabel("Rotation (rad)")
        plt.xlabel("Sample")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/robot_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved as robot_plot.png ({len(self.data)} samples)")


if __name__ == "__main__":
    robot = URController("192.168.1.33")

    robot.go_home()
    time.sleep(1)
    print(robot.get_state()["pose"])
    pose = robot.get_state()["pose"] + np.array([0, 0, -0.05, 0, 0, 0])
    print("Moving to pose:", pose)
    time.sleep(1)
    robot.moveL_ee(pose)
    time.sleep(1)

    robot.moveL_world([0, 0.0, 0.5, 0.0, 0.0, 0.0])

    robot.wait_for_commands()
    robot.wait_until_done()

    print("Sequence complete!")

    robot.disconnect()
    robot.plot()
