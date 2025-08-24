import time
import threading
import queue
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
import zmq

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robot_ipc_control.pose_estimation.transform_utils import (
    rvec_to_rotmat,
    rotmat_to_rvec,
)


class URController(threading.Thread):
    def __init__(self, ip, hz=50):
        super().__init__(daemon=True)
        self.ip = ip

        if self.ip == "192.168.1.66":
            self.robot_id = 1
            self.robot_config = np.load(
                "/home/weini/code/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_right.npy"
            )
            self.port = 5559
            self.ee2marker_offset = np.array([0.00, 0.05753, -0.10, 0, 0, 0])

        elif self.ip == "192.168.1.33":
            self.robot_id = 0
            self.robot_config = np.load(
                "/home/weini/code/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_left.npy"
            )
            self.port = 5556
            self.ee2marker_offset = np.array([0.00, -0.05753, -0.10, 0, 0, 0])
        else:
            self.robot_id = None
            self.robot_config = None
            self.port = None
            self.ee2marker_offset = None

        # Robot connections
        self.rtde_control = RTDEControlInterface(self.ip)
        self.rtde_receive = RTDEReceiveInterface(self.ip)

        # Command queue for threading
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()

        # State publisher
        self.hz = hz
        self.publishing = False
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://127.0.0.1:{self.port}")
        # self._start_publishing()

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
        self.default_speed = 1.0
        self.default_acceleration = 0.5
        self.default_joint_speed = 1.0
        self.default_joint_acceleration = 1.4

        # Data recording
        self.previous_force = None
        self.previous_force_world = None
        self.alpha = 1
        self.data = []
        self.forces = []

        # Start the control thread
        self.start()

    def run(self):
        """Main control thread loop - processes movement commands"""

        while not self._stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)

                # recording_thread = threading.Thread(
                #     target=self._record_during_movement, daemon=True
                # )
                # recording_thread.start()

                command()

                self.command_queue.task_done()
            except queue.Empty:
                continue

    def _start_publishing(self):
        """Start the state publishing thread"""

        self.publishing = True
        self.publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher_thread.start()

    def _record_during_movement(self):
        """Records robot state data while movement is happening"""
        time.sleep(0.1)

        for _ in range(100):
            state = self.get_state()
            self.data.append(state)
            time.sleep(0.05)

            if not self.is_moving():
                break

    def _publish_loop(self):
        """Main publishing loop - runs in background thread"""

        interval = 1.0 / self.hz

        def to_list(data):
            """Convert numpy arrays to lists for JSON serialization"""
            if hasattr(data, "tolist"):
                return data.tolist()
            elif isinstance(data, (list, tuple)):
                return list(data)
            else:
                return data

        while self.publishing:
            try:
                start_time = time.time()

                robot_state = self.get_state()

                if robot_state and self.publisher:
                    # convert np arrays to lists for JSON serialization
                    message = {
                        "timestamp": time.time(),
                        "robot_id": self.robot_id,
                        "Q": to_list(robot_state.get("joints", [])),
                        "pos": to_list(robot_state.get("pose", [])),
                        "vel": to_list(robot_state.get("speed", [])),
                        "force": to_list(robot_state.get("force", [])),
                        "filtered_force": to_list(
                            robot_state.get("filtered_force", [])
                        ),
                        "pose_world": to_list(robot_state.get("pose_world", [])),
                        "vel_world": to_list(robot_state.get("speed_world", [])),
                        "force_world": to_list(robot_state.get("force_world", [])),
                        "filtered_force_world": to_list(
                            robot_state.get("filtered_force_world", [])
                        ),
                        "gripper_world": to_list(robot_state.get("gripper_world", [])),
                    }

                    self.publisher.send_json(message)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Publishing error for {self.robot_id}: {e}")
                time.sleep(0.1)

    def world_2_robot(self, world_pose, robot_to_world):
        world_pose = np.array(world_pose)

        world_T = np.eye(4)
        world_T[:3, 3] = world_pose[:3]
        world_T[:3, :3] = rvec_to_rotmat(world_pose[3:])

        world_to_robot = np.linalg.inv(robot_to_world)
        robot_T = world_to_robot @ world_T

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

    def robot_2_world(self, local_pose, robot_to_world):
        local_pose_6d = np.array(local_pose)

        local_T = np.eye(4)
        local_T[:3, 3] = local_pose_6d[:3]
        local_T[:3, :3] = rvec_to_rotmat(local_pose_6d[3:])

        world_T = robot_to_world @ local_T

        world_pos = world_T[:3, 3]
        world_rvec = rotmat_to_rvec(world_T[:3, :3])

        return np.array(
            [
                world_pos[0],
                world_pos[1],
                world_pos[2],
                world_rvec[0],
                world_rvec[1],
                world_rvec[2],
            ]
        )

    def moveL(self, pose_ee):
        command = lambda: self.rtde_control.moveL(
            pose_ee, self.default_speed, self.default_acceleration
        )
        self.command_queue.put(command)

    def moveL_world(self, pose_world):
        if isinstance(pose_world, np.ndarray):
            pose_world = pose_world.tolist()

        try:
            pose_ee = self.world_2_robot(pose_world, self.robot_config)

            self.moveL(pose_ee)

        except Exception as e:
            print(f"ERROR in moveL_world: {e}")

    def moveL_gripper_world(self, TCP_world):
        TCP_world = np.array(TCP_world)

        try:
            gripper_pose = TCP_world + self.ee2marker_offset

            self.moveL_world(gripper_pose)

        except Exception as e:
            print(f"ERROR in moveL_gripper_world: {e}")

    def speedL(self, speed_ee, acceleration=0.5, time_duration=0.1):
        command = lambda: self.rtde_control.speedL(
            speed_ee, acceleration, time_duration
        )
        self.command_queue.put(command)

    def speedL_world(self, speed_world, acceleration=0.5, time_duration=0.1):
        if isinstance(speed_world, np.ndarray):
            speed_world = speed_world.tolist()

        try:
            vel_world = np.array(speed_world[:3])
            omega_world = np.array(speed_world[3:6])

            world_to_robot = np.linalg.inv(self.robot_config)
            robot_rotation = world_to_robot[:3, :3]

            vel_robot_base = robot_rotation @ vel_world
            omega_robot_base = omega_world

            speed_command = [
                vel_robot_base[0],
                vel_robot_base[1],
                vel_robot_base[2],
                omega_robot_base[0],
                omega_robot_base[1],
                omega_robot_base[2],
            ]

            self.speedL(
                speed_command, acceleration=acceleration, time_duration=time_duration
            )

        except Exception as e:
            print(f"ERROR in speedL_world: {e}")

    def speedStop(self):
        command = lambda: self.rtde_control.speedStop()
        self.command_queue.put(command)

    def moveJ(self, joints):
        command = lambda: self.rtde_control.moveJ(
            joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self.command_queue.put(command)

    def go_home(self):
        command = lambda: self.rtde_control.moveJ(
            self.home_joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self.command_queue.put(command)

    def force_low_pass_filter(self, previous_force, current_force, alpha):
        if previous_force is None:
            return current_force
        return (1 - alpha) * current_force + alpha * previous_force

    def get_state(self):
        """Get robot state with detailed timing breakdown"""

        pose_robot_base = self.rtde_receive.getActualTCPPose()

        force_robot_base = np.array(self.rtde_receive.getActualTCPForce())

        speed_robot_base = np.array(self.rtde_receive.getActualTCPSpeed())

        joints = self.rtde_receive.getActualQ()

        # === COORDINATE TRANSFORMATIONS ===
        pose_world = self.robot_2_world(pose_robot_base, self.robot_config)

        rotation_robot_to_world = self.robot_config[:3, :3]
        linear_speed_world = rotation_robot_to_world @ speed_robot_base[:3]
        angular_speed_world = rotation_robot_to_world @ speed_robot_base[3:]
        speed_world = np.concatenate((linear_speed_world, angular_speed_world))

        force_vector_world = rotation_robot_to_world @ force_robot_base[:3]
        torque_vector_world = rotation_robot_to_world @ force_robot_base[3:]
        force_world = np.concatenate((force_vector_world, torque_vector_world))

        # === FILTERING ===
        filtered_force = self.force_low_pass_filter(
            self.previous_force, force_robot_base, self.alpha
        )
        self.previous_force = filtered_force

        filtered_force_world = self.force_low_pass_filter(
            self.previous_force_world, force_vector_world, self.alpha
        )
        self.previous_force_world = filtered_force_world

        # === FINAL CALCULATIONS ===
        gripper_world = pose_world - self.ee2marker_offset

        self.forces.append([force_vector_world, filtered_force_world])

        return {
            "pose": pose_robot_base,
            "pose_world": pose_world,
            "joints": joints,
            "speed": speed_robot_base,
            "speed_world": speed_world,
            "force": force_robot_base,
            "force_world": force_world,
            "filtered_force": filtered_force,
            "filtered_force_world": filtered_force_world,
            "gripper_world": gripper_world,
        }

    def plot_forces(self):
        """
        Plot both raw and filtered forces over time.
        Assumes self.forces contains [force_world, filtered_force_world] pairs.
        """
        if not self.forces:
            print("No force data to plot")
            return

        # Extract force data
        forces_array = np.array(self.forces)
        raw_forces = forces_array[:, 0]  # force_world
        filtered_forces = forces_array[:, 1]  # filtered_force_world

        # Create time array (assuming constant sampling rate)
        time = np.arange(len(raw_forces))

        # Create subplots for each force component (assuming 3D forces: x, y, z)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(
            f"Force Comparison: Raw vs Filtered, Alpha: {self.alpha}", fontsize=16
        )

        labels = ["X", "Y", "Z"]
        colors_raw = ["red", "green", "blue"]
        colors_filtered = ["darkred", "darkgreen", "darkblue"]

        for i in range(3):
            ax = axes[i]

            # Plot raw forces
            raw_force_component = [f[i] for f in raw_forces]
            ax.plot(
                time,
                raw_force_component,
                color=colors_raw[i],
                alpha=0.6,
                linewidth=1,
                label=f"Raw Force {labels[i]}",
            )

            # Plot filtered forces
            filtered_force_component = [f[i] for f in filtered_forces]
            ax.plot(
                time,
                filtered_force_component,
                color=colors_filtered[i],
                linewidth=2,
                label=f"Filtered Force {labels[i]}",
            )

            ax.set_ylabel(f"Force {labels[i]} (N)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()

        # Save the plot instead of showing
        plt.savefig("force_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Force comparison plot saved as 'force_comparison.png'")

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
    robotR = URController("192.168.1.66")
    robotL = URController("192.168.1.33")

    time.sleep(0.1)

    try:
        robotR.go_home()
        robotL.go_home()

        robotR.wait_for_commands()
        robotL.wait_for_commands()

        robotR.wait_until_done()
        robotL.wait_until_done()

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robotR.disconnect()
        robotL.disconnect()
