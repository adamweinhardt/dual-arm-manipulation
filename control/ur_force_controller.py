import time
import threading
import numpy as np
import zmq
import os
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

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=0.2, dt=0.02):
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


class VectorPIDController:
    """3D Vector PID Controller - separate PID for each axis"""

    def __init__(self, kp, ki, kd):
        """
        Args:
            kp, ki, kd: Can be scalars (same gains for all axes) or 3-element arrays
        """
        self.kp = np.array(kp) if not np.isscalar(kp) else np.array([kp, kp, kp])
        self.ki = np.array(ki) if not np.isscalar(ki) else np.array([ki, ki, ki])
        self.kd = np.array(kd) if not np.isscalar(kd) else np.array([kd, kd, kd])

        self.integral = np.zeros(3)
        self.last_error = None
        self.last_time = None

    def update(self, error_vector):
        """
        Args:
            error_vector: 3D numpy array [ex, ey, ez]
        Returns:
            output_vector: 3D numpy array [ux, uy, uz]
        """
        error_vector = np.array(error_vector)
        current_time = time.time()

        if self.last_time is None:
            dt = 0.01
            derivative = np.zeros(3)
        else:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = 0.01

            if self.last_error is not None:
                derivative = (error_vector - self.last_error) / dt
            else:
                derivative = np.zeros(3)

        self.integral += error_vector * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)

        output = self.kp * error_vector + self.ki * self.integral + self.kd * derivative

        self.last_error = error_vector.copy()
        self.last_time = current_time

        return output

    def reset(self):
        """Reset PID state"""
        self.integral = np.zeros(3)
        self.last_error = None
        self.last_time = None


class URForceController(URController):
    """URController with force control capabilities"""

    def __init__(
        self, ip, hz=50, kp_f=0.01, ki_f=0, kd_f=0, kp_p=0.01, ki_p=0, kd_p=0.0
    ):
        super().__init__(ip)

        self.control_active = False
        self.control_thread = None
        self.control_stop = threading.Event()

        self.force_pid = VectorPIDController(kp=kp_f, ki=ki_f, kd=kd_f)
        self.pose_pid = VectorPIDController(kp=kp_p, ki=ki_p, kd=kd_p)

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

        if self.robot_id == 0:
            grasping_point = grasping_info.get("grasping_point0")
            approach_point = grasping_info.get("approach_point0")
            normal_vector = grasping_info.get("normal0")
        elif self.robot_id == 1:
            grasping_point = grasping_info.get("grasping_point1")
            approach_point = grasping_info.get("approach_point1")
            normal_vector = grasping_info.get("normal1")
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
        self.start_position = np.array(current_state["pose_world"][:3])

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
        self.control_thread = threading.Thread(target=self._control_loop3D, daemon=True)
        self.control_thread.start()

    def control_to_target(self, reference_force=None, distance_cap=0.2, timeout=30.0):
        if self.control_active:
            print("Control already active!")
            return False

        target_position, _, normal = self.get_grasping_data()

        direction = -np.array(normal, dtype=float)  # into the surface

        direction = direction / np.linalg.norm(direction)

        current_state = self.get_state()
        self.start_position = np.array(current_state["pose_world"][:3])

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
        self.control_thread = threading.Thread(target=self._control_loop3D, daemon=True)
        self.control_thread.start()

    def _control_loop3D(self):
        """Main 3D vector PID control loop"""
        control_period = 1.0 / self.control_rate_hz

        while self.control_active and not self.control_stop.is_set():
            loop_start = time.time()

            try:
                current_state = self.get_state()

                current_force_vector = np.array(
                    current_state["filtered_force_world"][:3]
                )
                current_position = np.array(current_state["gripper_world"][:3])

                # === 3D FORCE CONTROL ===
                if np.isscalar(self.ref_force):
                    ref_force_vector = self.ref_force * (
                        -self.control_direction
                    )  # TODO check the sign
                else:
                    ref_force_vector = np.array(self.ref_force)

                force_error_vector = ref_force_vector - current_force_vector
                force_output_vector = self.force_pid.update(force_error_vector)

                # === 3D POSITION CONTROL ===
                position_error_vector = self.target_position - current_position
                position_output_vector = self.pose_pid.update(position_error_vector)

                # === COMBINE 3D OUTPUTS ===
                total_output_vector = -force_output_vector + position_output_vector

                speed_command = [
                    total_output_vector[0],
                    total_output_vector[1],
                    total_output_vector[2],
                    0,
                    0,
                    0,
                ]

                # Data logging
                data_point = {
                    "timestamp": time.time() - self.start_time,
                    "force_vector": current_force_vector.copy(),
                    "position": current_position.copy(),
                    "target_position": self.target_position.copy(),
                    "reference_force_vector": ref_force_vector.copy(),
                    "force_error_vector": force_error_vector.copy(),
                    "position_error_vector": position_error_vector.copy(),
                    "force_output_vector": force_output_vector.copy(),
                    "position_output_vector": position_output_vector.copy(),
                    "total_output_vector": total_output_vector.copy(),
                }
                self.control_data.append(data_point)

                # Safety checks
                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break

                if time.time() - self.start_time >= self.control_timeout:
                    print(f"Control timeout reached")
                    break

                self.speedL_world(speed_command, acceleration=0.1, time_duration=0.1)

                # # Debug output (every 1 second)
                # if int(time.time() * 1) % 1 == 0 and int(time.time() * 10) % 10 == 0:
                #     print("-------")
                #     print(f"Reference force (scalar): {self.ref_force}")
                #     print(f"Control direction (negated): {-self.control_direction}")
                #     print(f"Reference force vector: {ref_force_vector}")
                #     print(f"Current force vector: {current_force_vector}")
                #     print(f"Force error vector: {force_error_vector}")
                #     print(f"Force PID output vector: {force_output_vector}")

                #     print(f"Current position: {current_position}")
                #     print(f"Target position: {self.target_position}")
                #     print(f"Position error vector: {position_error_vector}")
                #     print(f"Position PID output vector: {position_output_vector}")

                #     print(
                #         f"Total output vector (force + position): {total_output_vector}"
                #     )
                #     print(f"Speed command [vx, vy, vz, wx, wy, wz]: {speed_command}")
                #     print("-------")

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
        print("3D vector control loop ended")

    def _control_loop(self):
        """Main dual PID control loop"""
        control_period = 1.0 / self.control_rate_hz

        while self.control_active and not self.control_stop.is_set():
            loop_start = time.time()

            try:
                current_state = self.get_state()

                current_force_vector = np.array(current_state["filtered_force"][:3])
                current_position = np.array(current_state["pose_world"][:3])
                # Log data

                # === FORCE CONTROL ===
                ref_force_in_direction = self.ref_force * -self.control_direction
                force_error = ref_force_in_direction - current_force_vector
                force_output = self.force_pid.update(force_error)

                print("-------")
                print(f"Reference force vector: {self.ref_force}")
                print(f"Control direction (negated): {self.control_direction}")
                print(f"Reference force in control direction: {ref_force_in_direction}")
                print(f"Current force vector: {current_force_vector}")
                print(f"Force error: {force_error}")

                # === POSITION CONTROL ===
                position_in_direction = np.dot(
                    current_position - self.target_position, self.control_direction
                )
                position_error = -position_in_direction
                position_output = self.pose_pid.update(position_error)

                print(f"Current position: {current_position}")
                print(f"Target position: {self.target_position}")
                print(
                    f"Position difference (current - target): {current_position - self.target_position}"
                )
                print(f"Control direction: {self.control_direction}")
                print(
                    f"Position displacement in control direction: {position_in_direction}"
                )
                print(f"Position error: {position_error}")

                total_output = force_output + position_output

                velocity = self.control_direction * total_output
                speed_command = [velocity[0], velocity[1], velocity[2], 0, 0, 0]

                print(f"Position PID output: {position_output}")
                print(f"Force PID output: {force_output}")
                print(f"Total output (force + position): {total_output}")
                print(f"Velocity vector: {velocity}")
                print(f"Speed command [vx, vy, vz, wx, wy, wz]: {speed_command}")
                print("-------")

                data_point = {
                    "timestamp": time.time() - self.start_time,
                    "force_vector": current_force_vector.copy(),
                    "position": current_position.copy(),
                    "target_position": self.target_position.copy(),
                    "reference_force": self.ref_force,
                    "force_output": force_output,
                    "position_output": position_output,
                    "total_output": total_output,
                }
                self.control_data.append(data_point)

                # Safety checks
                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break

                if time.time() - self.start_time >= self.control_timeout:
                    print(f"Control timeout reached")
                    break

                self.speedL_world(speed_command, acceleration=0.1, time_duration=0.1)

                # Debug output (every 1 second)
                if int(time.time() * 1) % 1 == 0 and int(time.time() * 10) % 10 == 0:
                    print(
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

    def wait_for_control(self):
        """Wait for force control to complete"""
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join()

    def stop_control(self):
        """Stop the force control loop"""
        if self.control_active:
            self.control_active = False
            self.control_stop.set()
            if self.control_thread:
                self.control_thread.join(timeout=2.0)
            print("Control stopped")

    def disconnect(self):
        """Override disconnect to stop force control first"""
        self.stop_control()
        super().disconnect()

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

        force_errors_val = np.array(
            [ref - abs(f_dir) for ref, f_dir in zip(ref_forces, force_in_direction)]
        )

        position_errors_magnitude = np.linalg.norm(positions - target_positions, axis=1)

        total_errors = np.abs(force_errors_val) + np.abs(position_errors_magnitude)

        force_outputs = np.array([d["force_output"] for d in self.control_data])
        position_outputs = np.array([d["position_output"] for d in self.control_data])
        total_outputs = force_outputs + position_outputs

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        initial_target_pose_str = f"[{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]"

        force_pid_str = f"Kp={self.force_pid.kp:.3f}, Ki={self.force_pid.ki:.3f}, Kd={self.force_pid.kd:.3f}"
        pose_pid_str = f"Kp={self.pose_pid.kp:.3f}, Ki={self.pose_pid.ki:.3f}, Kd={self.pose_pid.kd:.3f}"

        plt.figure(figsize=(18, 10))

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
        plt.subplot(2, 3, 6)
        plt.plot(
            timestamps, force_outputs, label="Force Output", linewidth=2, color="red"
        )
        plt.plot(
            timestamps,
            position_outputs,
            label="Position Output",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps, total_outputs, label="Total Output", linewidth=2, color="purple"
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("Control Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Control Output")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust rect to make space for suptitle

        # Save filename with date
        filename = (
            f"plots/comprehensive_control_plot_{current_datetime}_{self.robot_id}.png"
        )
        os.makedirs("plots", exist_ok=True)
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

    def plot_data3D(self):
        """Generate comprehensive plots for control data including pose reference tracking"""
        if not self.control_data:
            print("No data to plot")
            return

        # Extract data from logged data
        timestamps = [d["timestamp"] for d in self.control_data]
        force_vectors = np.array([d["force_vector"] for d in self.control_data])
        positions = np.array([d["position"] for d in self.control_data])
        target_positions = np.array([d["target_position"] for d in self.control_data])

        # Updated vector data extraction
        reference_force_vectors = np.array(
            [d["reference_force_vector"] for d in self.control_data]
        )
        force_error_vectors = np.array(
            [d["force_error_vector"] for d in self.control_data]
        )
        position_error_vectors = np.array(
            [d["position_error_vector"] for d in self.control_data]
        )
        force_output_vectors = np.array(
            [d["force_output_vector"] for d in self.control_data]
        )
        position_output_vectors = np.array(
            [d["position_output_vector"] for d in self.control_data]
        )
        total_output_vectors = np.array(
            [d["total_output_vector"] for d in self.control_data]
        )

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
        reference_force_magnitudes = np.linalg.norm(reference_force_vectors, axis=1)
        force_in_direction = [
            np.dot(fv, self.control_direction) for fv in force_vectors
        ]
        ref_force_in_direction = [
            np.dot(rfv, self.control_direction) for rfv in reference_force_vectors
        ]

        distances_from_start = np.linalg.norm(positions - self.start_position, axis=1)

        # Position errors in control direction
        position_errors_in_direction = [
            np.dot(pos_err, self.control_direction)
            for pos_err in position_error_vectors
        ]

        # Force errors in control direction
        force_errors_in_direction = [
            np.dot(force_err, self.control_direction)
            for force_err in force_error_vectors
        ]

        # Error magnitudes
        force_error_magnitudes = np.linalg.norm(force_error_vectors, axis=1)
        position_error_magnitudes = np.linalg.norm(position_error_vectors, axis=1)

        # Output magnitudes
        force_output_magnitudes = np.linalg.norm(force_output_vectors, axis=1)
        position_output_magnitudes = np.linalg.norm(position_output_vectors, axis=1)
        total_output_magnitudes = np.linalg.norm(total_output_vectors, axis=1)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        initial_target_pose_str = f"[{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]"

        # PID parameters (handle both scalar and vector PIDs)
        if hasattr(self.force_pid, "kp") and np.isscalar(self.force_pid.kp):
            force_pid_str = f"Kp={self.force_pid.kp:.3f}, Ki={self.force_pid.ki:.3f}, Kd={self.force_pid.kd:.3f}"
        else:
            force_pid_str = f"Kp={self.force_pid.kp}, Ki={self.force_pid.ki}, Kd={self.force_pid.kd}"

        if hasattr(self.pose_pid, "kp") and np.isscalar(self.pose_pid.kp):
            pose_pid_str = f"Kp={self.pose_pid.kp:.3f}, Ki={self.pose_pid.ki:.3f}, Kd={self.pose_pid.kd:.3f}"
        else:
            pose_pid_str = (
                f"Kp={self.pose_pid.kp}, Ki={self.pose_pid.ki}, Kd={self.pose_pid.kd}"
            )

        plt.figure(figsize=(18, 12))

        # Main title for the entire figure
        ref_force_str = (
            f"{reference_force_magnitudes[0]:.2f}N"
            if len(reference_force_magnitudes) > 0
            else "N/A"
        )
        plt.suptitle(
            f"3D Vector Control Data Plot ({current_datetime})\n"
            f"Ref Force: {ref_force_str}, Target Pose: {initial_target_pose_str}\n"
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
            fontsize=16,
            y=0.98,
        )

        # === ROW 1: Reference Tracking and Movement ===

        # 1. Force vs Target Force (3D Components)
        plt.subplot(3, 3, 1)
        plt.plot(
            timestamps, force_vectors[:, 0], label="X Force", linewidth=2, color="red"
        )
        plt.plot(
            timestamps, force_vectors[:, 1], label="Y Force", linewidth=2, color="green"
        )
        plt.plot(
            timestamps, force_vectors[:, 2], label="Z Force", linewidth=2, color="blue"
        )
        plt.plot(
            timestamps,
            reference_force_vectors[:, 0],
            label="X Target",
            linewidth=2,
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            reference_force_vectors[:, 1],
            label="Y Target",
            linewidth=2,
            linestyle="--",
            color="green",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            reference_force_vectors[:, 2],
            label="Z Target",
            linewidth=2,
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.title("Force vs Target Force")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Position vs Target Position
        plt.subplot(3, 3, 2)
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

        # 3. Movement Distance vs Time
        plt.subplot(3, 3, 3)
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

        # 4. Force Error Vectors (3D)
        plt.subplot(3, 3, 4)
        plt.plot(
            timestamps,
            force_error_vectors[:, 0],
            label="Force Error X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            force_error_vectors[:, 1],
            label="Force Error Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            force_error_vectors[:, 2],
            label="Force Error Z",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps,
            force_error_magnitudes,
            label="Force Error Magnitude",
            linewidth=2,
            color="black",
            linestyle="--",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Force Error Vectors")
        plt.xlabel("Time (s)")
        plt.ylabel("Force Error (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Position Error Vectors (3D)
        plt.subplot(3, 3, 5)
        plt.plot(
            timestamps,
            position_error_vectors[:, 0],
            label="Position Error X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            position_error_vectors[:, 1],
            label="Position Error Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            position_error_vectors[:, 2],
            label="Position Error Z",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps,
            position_error_magnitudes,
            label="Position Error Magnitude",
            linewidth=2,
            color="black",
            linestyle="--",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Position Error Vectors")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Directional Errors (along control direction)
        plt.subplot(3, 3, 6)
        plt.plot(
            timestamps,
            force_errors_in_direction,
            label="Force Error in Direction",
            linewidth=2,
            color="darkred",
        )
        plt.plot(
            timestamps,
            position_errors_in_direction,
            label="Position Error in Direction",
            linewidth=2,
            color="darkgreen",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("Errors in Control Direction")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # === ROW 3: Control Outputs ===

        # 7. Force Output Vectors (3D)
        plt.subplot(3, 3, 7)
        plt.plot(
            timestamps,
            force_output_vectors[:, 0],
            label="Force Output X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            force_output_vectors[:, 1],
            label="Force Output Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            force_output_vectors[:, 2],
            label="Force Output Z",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps,
            force_output_magnitudes,
            label="Force Output Magnitude",
            linewidth=2,
            color="black",
            linestyle="--",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Force Control Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Force Output")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 8. Position Output Vectors (3D)
        plt.subplot(3, 3, 8)
        plt.plot(
            timestamps,
            position_output_vectors[:, 0],
            label="Position Output X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            position_output_vectors[:, 1],
            label="Position Output Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            position_output_vectors[:, 2],
            label="Position Output Z",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps,
            position_output_magnitudes,
            label="Position Output Magnitude",
            linewidth=2,
            color="black",
            linestyle="--",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Position Control Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Output")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 9. Total Output Vectors (3D)
        plt.subplot(3, 3, 9)
        plt.plot(
            timestamps,
            total_output_vectors[:, 0],
            label="Total Output X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            total_output_vectors[:, 1],
            label="Total Output Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            total_output_vectors[:, 2],
            label="Total Output Z",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            timestamps,
            total_output_magnitudes,
            label="Total Output Magnitude",
            linewidth=2,
            color="purple",
            linestyle="--",
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Total Control Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Output")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save filename with date
        filename = f"plots/comprehensive_3d_control_plot_{current_datetime}_{self.robot_id}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        # Print comprehensive summary statistics
        if timestamps:
            final_force_mag = force_magnitudes[-1] if len(force_magnitudes) > 0 else 0
            final_ref_force_mag = (
                reference_force_magnitudes[-1]
                if len(reference_force_magnitudes) > 0
                else 0
            )
            final_force_in_dir = force_in_direction[-1] if force_in_direction else 0
            final_ref_force_in_dir = (
                ref_force_in_direction[-1] if ref_force_in_direction else 0
            )

            max_distance = (
                max(distances_from_start) if len(distances_from_start) > 0 else 0
            )
            control_duration = timestamps[-1]

            final_position_error_mag = (
                position_error_magnitudes[-1]
                if len(position_error_magnitudes) > 0
                else 0
            )
            final_position_error_dir = (
                position_errors_in_direction[-1] if position_errors_in_direction else 0
            )

            avg_force_error_mag = np.mean(force_error_magnitudes)
            avg_position_error_mag = np.mean(position_error_magnitudes)
            avg_force_error_dir = np.mean(np.abs(force_errors_in_direction))
            avg_position_error_dir = np.mean(np.abs(position_errors_in_direction))

            print("=" * 70)
            print("COMPREHENSIVE 3D VECTOR CONTROL SUMMARY")
            print("=" * 70)
            print(f"Plot saved as: {filename} ({len(self.control_data)} samples)")
            print(f"Control duration: {control_duration:.2f}s")
            print("")
            print("FORCE TRACKING:")
            print(f"  Final force magnitude: {final_force_mag:.3f}N")
            print(f"  Reference force magnitude: {final_ref_force_mag:.3f}N")
            print(f"  Final force in direction: {final_force_in_dir:.3f}N")
            print(f"  Reference force in direction: {final_ref_force_in_dir:.3f}N")
            print(
                f"  Force error magnitude: {abs(final_ref_force_mag - final_force_mag):.3f}N"
            )
            print(f"  Average force error magnitude: {avg_force_error_mag:.3f}N")
            print(f"  Average force error in direction: {avg_force_error_dir:.3f}N")
            print("")
            print("POSITION TRACKING:")
            print(f"  Max distance moved: {max_distance:.3f}m")
            if hasattr(self, "distance_cap"):
                print(f"  Distance cap: {self.distance_cap}m")
            print(f"  Final position error magnitude: {final_position_error_mag:.3f}m")
            print(
                f"  Final position error in direction: {final_position_error_dir:.3f}m"
            )
            print(f"  Average position error magnitude: {avg_position_error_mag:.3f}m")
            print(
                f"  Average position error in direction: {avg_position_error_dir:.3f}m"
            )
            print("")
            print("CONTROL PARAMETERS:")
            print(
                f"  Control direction: [{self.control_direction[0]:.2f}, {self.control_direction[1]:.2f}, {self.control_direction[2]:.2f}]"
            )
            print(f"  Force PID: {force_pid_str}")
            print(f"  Pose PID: {pose_pid_str}")
            print("")
            print("FINAL OUTPUT MAGNITUDES:")
            print(f"  Force output: {force_output_magnitudes[-1]:.6f}")
            print(f"  Position output: {position_output_magnitudes[-1]:.6f}")
            print(f"  Total output: {total_output_magnitudes[-1]:.6f}")
            print("=" * 70)
        else:
            print("No data available for summary statistics")


if __name__ == "__main__":
    hz = 50

    kp_f = 0.0016
    ki_f = 0.0000
    kd_f = 0.00075

    kp_p = 0.3
    ki_p = 0.0
    kd_p = 0.001

    alpha = 0.2

    robotL = URForceController(
        "192.168.1.33",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
    )

    robotR = URForceController(
        "192.168.1.66",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
    )
    robotL.alpha = alpha
    robotR.alpha = alpha

    try:
        robotR.go_home()
        robotL.go_home()

        robotR.wait_for_commands()
        robotL.wait_for_commands()

        robotR.wait_until_done()
        robotL.wait_until_done()

        robotR.go_to_approach()
        robotL.go_to_approach()

        robotR.wait_for_commands()
        robotL.wait_for_commands()
        robotR.wait_until_done()
        robotL.wait_until_done()

        time.sleep(0.1)

        robotR.control_to_target(
            reference_force=10.0,
            distance_cap=0.30,
            timeout=15.0,
        )

        robotL.control_to_target(
            reference_force=10.0,
            distance_cap=0.30,
            timeout=15.0,
        )

        robotR.wait_for_control()
        robotL.wait_for_control()

        robotR.plot_data3D()
        robotL.plot_data3D()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robotR.stop_control()
        robotL.stop_control()
    except Exception as e:
        print(f"An error occurred: {e}")
        robotR.stop_control()
        robotL.stop_control()
    finally:
        robotR.disconnect()
        robotL.disconnect()
        print("Robot disconnected")
