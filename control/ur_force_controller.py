import time
import threading
import numpy as np
import zmq
import datetime
import matplotlib.pyplot as plt
from ur_controller import URController
from dual_arm_controller import DualArmController
from robot_ipc_control.pose_estimation.transform_utils import (
    rvec_to_rotmat,
    rotmat_to_rvec,
    end_effector_rotation_from_normal,
)


class PIDController:
    """Basic PID controller for force control"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=0.5, dt=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.dt = dt

        # states
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = self.dt

        p_term = self.kp * error

        self.integral += error * dt
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        output = np.clip(output, -self.max_output, self.max_output)

        self.prev_error = error
        self.last_time = current_time

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class URForceController(URController):
    """URController with force control capabilities"""

    def __init__(
        self, ip, hz=50, kp_f=0.01, ki_f=0, kd_f=0, kp_p=0.01, ki_p=0, kd_p=0.0
    ):
        super().__init__(ip)

        self.control_active = False
        self.control_thread = None
        self.control_stop = threading.Event()

        self.force_pid = PIDController(kp=kp_f, ki=ki_f, kd=kd_f)
        self.pose_pid = PIDController(kp=kp_p, ki=ki_p, kd=kd_p)

        self.control_rate_hz = hz
        self.min_force_threshold = 0.5

        self.control_data = []

        self.grasping_context = zmq.Context()
        self.grasping_socket = self.grasping_context.socket(zmq.SUB)
        self.grasping_socket.setsockopt(zmq.CONFLATE, 1)
        self.grasping_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        grasping_port = 5560
        self.grasping_socket.connect(f"tcp://127.0.0.1:{grasping_port}")

        self.current_grasping_data = {}

    def _update_grasping_data(self):
        """Update grasping data from ZMQ (non-blocking)"""
        try:
            message = self.grasping_socket.recv_json(flags=zmq.NOBLOCK)

            if "grasping_points" in message:
                self.current_grasping_data = message["grasping_points"]

                return True
            else:
                return False
        except zmq.Again:
            return False
        except Exception as e:
            print(f"Error receiving grasping data: {e}")
            return False

    def get_grasping_data(self):
        """Get approach point for this robot from latest grasping data"""

        # Wait for first message
        while not self._update_grasping_data():
            time.sleep(0.01)

        box_id = list(self.current_grasping_data.keys())[0]
        grasping_info = self.current_grasping_data[box_id]

        if self.robot_id == 1:
            grasping_point = grasping_info.get("grasping_point1")
            approach_point = grasping_info.get("approach_point1")
            normal_vector = grasping_info.get("normal1")
        elif self.robot_id == 0:
            grasping_point = grasping_info.get("grasping_point2")
            approach_point = grasping_info.get("approach_point2")
            normal_vector = grasping_info.get("normal2")
        else:
            return None, None

        if approach_point is None or normal_vector is None:
            return None, None

        return (
            np.array(grasping_point),
            np.array(approach_point),
            np.array(normal_vector),
        )

    def go_to_approach(self):
        """Go to approach point for grasping - simplified using moveL_world"""
        _, approach_point, normal = self.get_grasping_data()
        if approach_point is None:
            print("Cannot go to approach: no valid approach point")
            return False

        rotation_matrix = end_effector_rotation_from_normal(-normal)

        world_rotation = rotmat_to_rvec(rotation_matrix)

        print(f"Normal vector: {normal}")  # Add this
        print(f"Normal magnitude: {np.linalg.norm(normal)}")
        print(f"Rotation matrix:\n{rotation_matrix}")
        print(f"World rotation (rvec): {world_rotation}")

        world_pose = [
            approach_point[0],
            approach_point[1],
            approach_point[2],
            world_rotation[0],
            world_rotation[1],
            world_rotation[2],
        ]

        self.moveL_gripper_world(world_pose)

    def control_to_target_manual(
        self,
        target_position=None,
        reference_force=None,
        direction=[0, 0, -1],
        distance_cap=0.2,
        timeout=30.0,
    ):
        if self.control_active:
            print("Control already active!")
            return False

        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        current_state = self.get_state()
        self.start_position = np.array(current_state["pose"][:3])

        if target_position is None:
            self.target_position = self.start_position.copy()
        else:
            self.target_position = np.array(target_position)

        if reference_force is None:
            self.ref_force = 0.0
        else:
            self.ref_force = reference_force

        self.control_direction = direction
        self.distance_cap = distance_cap
        self.control_timeout = timeout
        self.start_time = time.time()

        self.force_pid.reset()
        self.pose_pid.reset()
        self.control_data = []

        self.rtde_control.zeroFtSensor()

        self.control_active = True
        self.control_stop.clear()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        print(f"Dual control started:")
        print(f"  Target position: {self.target_position}")
        print(f"  Reference force: {self.ref_force}N")
        print(f"  Direction: {direction}")
        print(f"  Distance cap: {distance_cap}m, Timeout: {timeout}s")

        return True

    def control_to_target(
        self,
        reference_force=None,
        distance_cap=0.2,
        timeout=30.0,
    ):
        if self.control_active:
            print("Control already active!")
            return False

        target_position, _, normal = self.get_grasping_data()
        direction = np.array(-normal, dtype=float)
        direction = direction / np.linalg.norm(direction)

        current_state = self.get_state()
        self.start_position = np.array(currentarget(
            reference_force=10.0,
            distance_cap=1_state["pose"][:3])

        if target_position is None:
            self.target_position = self.start_position.copy()
        else:
            self.target_position = np.array(target_position)

        if reference_force is None:
            self.ref_force = 0.0
        else:
            self.ref_force = reference_force

        self.control_direction = direction
        self.distance_cap = distance_cap
        self.control_timeout = timeout
        self.start_time = time.time()

        self.force_pid.reset()
        self.pose_pid.reset()
        self.control_data = []

        self.rtde_control.zeroFtSensor()

        self.control_active = True
        self.control_stop.clear()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        print(f"Dual control started:")
        print(f"  Target position: {self.target_position}")
        print(f"  Reference force: {self.ref_force}N")
        print(f"  Direction: {direction}")
        print(f"  Distance cap: {distance_cap}m, Timeout: {timeout}s")

        return True

    def _control_loop(self):
        """Main dual PID control loop"""
        control_period = 1.0 / self.control_rate_hz

        while self.control_active and not self.control_stop.is_set():
            loop_start = time.time()

            try:
                current_state = self.get_state()

                if (
                    not current_state
                    or current_state["force"] is None
                    or current_state["pose"] is None
                ):
                    print("ERROR: Invalid robot state, stopping control")
                    break

                current_force_vector = np.array(current_state["force"][:3])
                current_position = np.array(current_state["pose"][:3])

                # Log data
                data_point = {
                    "timestamp": time.time() - self.start_time,
                    "force_vector": current_force_vector.copy(),
                    "position": current_position.copy(),
                    "target_position": self.target_position.copy(),
                    "reference_force": self.ref_force,
                }
                self.control_data.append(data_point)

                # === FORCE CONTROL ===
                force_in_direction = np.dot(
                    current_force_vector, self.control_direction
                )
                force_error = self.ref_force - abs(force_in_direction)
                force_output = self.force_pid.update(force_error)

                # === POSITION CONTROL ===
                position_in_direction = np.dot(
                    current_position - self.target_position, self.control_direction
                )
                position_error = -position_in_direction
                position_output = self.pose_pid.update(position_error)

                total_output = force_output + position_output

                velocity = self.control_direction * total_output
                speed_command = [velocity[0], velocity[1], velocity[2], 0, 0, 0]

                # Safety checks
                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break

                if time.time() - self.start_time >= self.control_timeout:
                    print(f"Control timeout reached")
                    break

                self.speedL(speed_command, acceleration=0.1, time_duration=0.1)

                # Debug output (every 1 second)
                if int(time.time() * 1) % 1 == 0 and int(time.time() * 10) % 10 == 0:
                    print(
                        f"Force: {force_in_direction:.2f}N (target: {self.ref_force}N), "
                        f"Pos error: {position_error:.3f}m, "
                        f"Combined output: {total_output:.3f}, "
                        f"Distance: {distance_moved:.3f}m"
                    )

            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.speedStop()
                except:
                    pass
                break

            # Maintain control rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, control_period - elapsed)
            time.sleep(sleep_time)

        # Clean stop
        try:
            self.speedStop()
        except:
            pass

        self.control_active = False
        print("Dual control loop ended")

    def stop_control(self):
        """Stop the force control loop"""
        if self.control_active:
            self.control_active = False
            self.control_stop.set()
            if self.control_thread:
                self.control_thread.join(timeout=2.0)
            print("Control stopped")

    def wait_for_control(self):
        """Wait for force control to complete"""
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join()

    def get_current_force(self):
        """Get current TCP force reading"""
        return self.get_state()["force"]

    def get_force_magnitude(self):
        """Get magnitude of current force vector"""
        force_vector = np.array(self.get_state()["force"][:3])
        return np.linalg.norm(force_vector)

    def plot_data(self):
        """Generate comprehensive plots for control data including pose reference tracking"""
        if not self.control_data:
            print("No data to plot")
            return

        # Extract data from logged data
        timestamps = [d["timestamp"] for d in self.control_data]
        force_vectors = np.array([d["force_vector"] for d in self.control_data])
        positions = np.array([d["position"] for d in self.control_data])
        target_positions = np.array([d["target_position"] for d in self.control_data])
        ref_forces = [d["reference_force"] for d in self.control_data]

        # Handle both 3D and 6D target positions (extract only position part)
        if target_positions.shape[1] == 6:
            target_positions = target_positions[
                :, :3
            ]  # Extract only xyz, ignore rotation
        elif target_positions.shape[1] != 3:
            print(
                f"Warning: Unexpected target_position shape {target_positions.shape}, using first 3 elements"
            )
            target_positions = target_positions[:, :3]

        # Calculate derived metrics
        force_magnitudes = np.linalg.norm(force_vectors, axis=1)
        force_in_direction = [
            np.dot(fv, self.control_direction) for fv in force_vectors
        ]
        distances_from_start = np.linalg.norm(positions - self.start_position, axis=1)

        # Position errors in control direction (now both are 3D)
        position_errors_in_direction = [
            np.dot(pos - target, self.control_direction)
            for pos, target in zip(positions, target_positions)
        ]

        # Force and Position Errors for Total Error Calculation
        force_errors_val = np.array(
            [ref - abs(f_dir) for ref, f_dir in zip(ref_forces, force_in_direction)]
        )

        # For simplicity in total error, let's consider the magnitude of the position error
        # (This can be adjusted based on how you define "total error")
        position_errors_magnitude = np.linalg.norm(positions - target_positions, axis=1)

        # Define "Total Error" - for demonstration, sum of absolute force error and position error magnitude
        # You might want to normalize or weight these errors based on their units and importance.
        total_errors = np.abs(force_errors_val) + np.abs(
            position_errors_magnitude
        )  # Example total error

        # --- Dynamic Plot Title Information ---
        current_datetime = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # Use the initial target_position for the plot title
        initial_target_pose_str = f"[{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]"

        force_pid_str = f"Kp={self.force_pid.kp:.3f}, Ki={self.force_pid.ki:.3f}, Kd={self.force_pid.kd:.3f}"
        pose_pid_str = f"Kp={self.pose_pid.kp:.3f}, Ki={self.pose_pid.ki:.3f}, Kd={self.pose_pid.kd:.3f}"

        # Create comprehensive plot (now 2 rows, 3 columns)
        plt.figure(figsize=(18, 10))  # Adjusted figure size for 2 rows

        # Main title for the entire figure
        plt.suptitle(
            f"Control Data Plot ({current_datetime})\n"
            f"Ref Force: {self.ref_force:.2f}N, Target Pose: {initial_target_pose_str}\n"
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
            fontsize=16,
            y=1.02,  # Adjust y to move title up
        )

        # === ROW 1: Reference Tracking and Movement ===

        # 1. Force Magnitude and Reference Tracking (formerly subplot 3,3,2)
        plt.subplot(2, 3, 1)  # Changed to 2 rows, 3 columns, plot 1
        plt.plot(
            timestamps,
            force_magnitudes,
            label="Force Magnitude",
            linewidth=2,
            color="purple",
        )
        plt.plot(
            timestamps,
            [abs(f) for f in force_in_direction],
            label="Force in Direction",
            linewidth=2,
            color="orange",
        )
        plt.plot(
            timestamps,
            ref_forces,
            label="Reference Force",
            linewidth=3,
            color="red",
            linestyle="--",
        )
        plt.title("Force Magnitude vs Reference")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Position vs Target Position (formerly subplot 3,3,4)
        plt.subplot(2, 3, 2)  # Changed to 2 rows, 3 columns, plot 2
        plt.plot(
            timestamps, positions[:, 0], label="X Position", linewidth=2, color="red"
        )
        plt.plot(
            timestamps, positions[:, 1], label="Y Position", linewidth=2, color="green"
        )
        plt.plot(
            timestamps, positions[:, 2], label="Z Position", linewidth=2, color="blue"
        )
        plt.plot(
            timestamps,
            target_positions[:, 0],
            label="X Target",
            linewidth=2,
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            target_positions[:, 1],
            label="Y Target",
            linewidth=2,
            linestyle="--",
            color="green",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            target_positions[:, 2],
            label="Z Target",
            linewidth=2,
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.title("Position vs Target Position")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Movement Distance vs Time (formerly subplot 3,3,6)
        plt.subplot(2, 3, 3)  # Changed to 2 rows, 3 columns, plot 3
        plt.plot(
            timestamps,
            distances_from_start,
            label="Distance from Start",
            linewidth=2,
            color="brown",
        )
        if hasattr(self, "distance_cap"):
            plt.axhline(
                y=self.distance_cap,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Distance Cap ({self.distance_cap}m)",
            )
        plt.title("Movement Distance vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # === ROW 2: Error Analysis ===

        # 4. Force Control Error (formerly subplot 3,3,3)
        plt.subplot(2, 3, 4)  # Changed to 2 rows, 3 columns, plot 4
        plt.plot(
            timestamps,
            force_errors_val,
            label="Force Error",
            linewidth=2,
            color="darkred",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        if hasattr(self, "min_force_threshold"):
            plt.axhline(
                y=self.min_force_threshold,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Threshold (±{self.min_force_threshold}N)",
            )
            plt.axhline(
                y=-self.min_force_threshold, color="orange", linestyle="--", linewidth=2
            )
        plt.title("Force Control Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Force Error (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Position Error in Control Direction (formerly subplot 3,3,5)
        plt.subplot(2, 3, 5)  # Changed to 2 rows, 3 columns, plot 5
        plt.plot(
            timestamps,
            position_errors_in_direction,
            label="Position Error in Control Direction",
            linewidth=2,
            color="darkgreen",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("Position Error in Control Direction")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Total Error Plot (NEW)
        plt.subplot(2, 3, 6)  # Changed to 2 rows, 3 columns, plot 6
        plt.plot(
            timestamps,
            total_errors,
            label="Total Error (Abs Force + Abs Pos)",
            linewidth=2,
            color="darkblue",
        )
        plt.title("Total Control Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Error")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust rect to make space for suptitle

        # Save filename with date
        filename = f"plots/comprehensive_control_plot_{current_datetime}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        # Print comprehensive summary statistics
        if timestamps:
            final_force = force_in_direction[-1] if force_in_direction else 0
            final_ref_force = ref_forces[-1] if ref_forces else 0
            max_distance = (
                max(distances_from_start) if len(distances_from_start) > 0 else 0
            )
            control_duration = timestamps[-1]
            final_position_error = (
                position_errors_in_direction[-1] if position_errors_in_direction else 0
            )

            avg_force_error = np.mean(
                np.abs(force_errors_val)
            )  # Use abs for average error
            avg_position_error = np.mean(
                np.abs(position_errors_in_direction)
            )  # Use abs for average error

            print("=" * 60)
            print("COMPREHENSIVE CONTROL SUMMARY")
            print("=" * 60)
            print(f"Plot saved as: {filename} ({len(self.control_data)} samples)")
            print(f"Control duration: {control_duration:.2f}s")
            print("")
            print("FORCE TRACKING:")
            print(f"  Final force: {final_force:.2f}N")
            print(f"  Reference force: {final_ref_force:.2f}N")
            print(f"  Force error: {abs(final_ref_force - abs(final_force)):.2f}N")
            print(f"  Average force error: {avg_force_error:.2f}N")
            print("")
            print("POSITION TRACKING:")
            print(f"  Max distance moved: {max_distance:.3f}m")
            if hasattr(self, "distance_cap"):
                print(f"  Distance cap: {self.distance_cap}m")
            print(
                f"  Final position error (in control dir): {final_position_error:.3f}m"
            )
            print(f"  Average position error: {avg_position_error:.3f}m")
            print("")
            print("CONTROL PARAMETERS:")
            print(
                f"  Control direction: [{self.control_direction[0]:.2f}, {self.control_direction[1]:.2f}, {self.control_direction[2]:.2f}]"
            )
            print(
                f"  Force PID: kp={self.force_pid.kp}, ki={self.force_pid.ki}, kd={self.force_pid.kd}"
            )
            print(
                f"  Pose PID: kp={self.pose_pid.kp}, ki={self.pose_pid.ki}, kd={self.pose_pid.kd}"
            )
            print("=" * 60)
        else:
            print("No data available for summary statistics")

    def disconnect(self):
        """Override disconnect to stop force control first"""
        self.stop_control()
        super().disconnect()


if __name__ == "__main__":
    hz = 50

    kp_f = 0.005
    ki_f = 0.000
    kd_f = 0.0001

    kp_p = 0.5
    ki_p = 0.2
    kd_p = 0.002

    robot1 = URForceController(
        "192.168.1.66",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
    )
    # robot2 = URForceController("192.168.1.33", hz=hz, kp=kp, ki=ki, kd=kd)

    try:
        robot1.go_to_approach()
        print("\nStarting force control...")
        robot1.control_to_target(
            reference_force=10.0,
            distance_cap=1,
            timeout=20.0,
        )

        # print("\nStarting force control...")
        # robot2.force_control_to_target(
        #     reference_force=8.0,reference_force
        #     direction=[0, 1, 0],
        #     distance_cap=0.2,
        #     timeout=20.0,
        # )

        robot1.wait_for_control()
        # # robot2.wait_for_force_control()

        # print("Force control complete!")
        # print("Final force reading:", robot1.get_current_force())

        robot1.plot_data()
        # robot2.plot_force_data()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot1.disconnect()
        # robot2.disconnect()
        print("Robot disconnected")
