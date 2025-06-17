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


class URController(threading.Thread):
    def __init__(self, ip):
        super().__init__(daemon=True)
        self.ip = ip

        # Command queue for threading
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()

        # Robot connections
        self.rtde_control = RTDEControlInterface(self.ip)
        self.rtde_receive = RTDEReceiveInterface(self.ip)

        # Defaults
        self.home_joints = [-pi / 2.0, -pi / 2.0, pi / 2.0, -pi / 2.0, -pi / 2.0, 0.0]
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

    def moveL(self, pose):
        command = lambda: self.rtde_control.moveL(
            pose, self.default_speed, self.default_acceleration
        )
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
        return {
            "pose": self.rtde_receive.getActualTCPPose(),
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
        plt.savefig("robot_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved as robot_plot.png ({len(self.data)} samples)")


if __name__ == "__main__":
    robot = URController("172.17.0.2")

    robot.go_home()
    robot.moveL([0.3, 0.2, 0.4, 0, 3.14, 0])
    robot.moveL([0.4, 0.3, 0.3, 0, 3.14, 0])

    robot.wait_for_commands()
    robot.wait_until_done()

    print("Sequence complete!")

    robot.disconnect()
    robot.plot()
