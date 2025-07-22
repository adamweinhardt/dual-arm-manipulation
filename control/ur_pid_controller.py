import time
import threading
import numpy as np
import zmq
import os
import datetime
import matplotlib.pyplot as plt
from ur_controller import URController
from robot_ipc_control.pose_estimation.transform_utils import (
    rotmat_to_rvec,
    end_effector_rotation_from_normal,
)


class VectorPIDController:
    """3D Vector PID Controller - separate PID for each axis"""

    def __init__(self, kp, ki, kd, dt=0.02):
        """
        Args:
            kp, ki, kd: Can be scalars (same gains for all axes) or 3-element arrays
        """
        self.kp = np.array(kp) if not np.isscalar(kp) else np.array([kp, kp, kp])
        self.ki = np.array(ki) if not np.isscalar(ki) else np.array([ki, ki, ki])
        self.kd = np.array(kd) if not np.isscalar(kd) else np.array([kd, kd, kd])

        self.dt = dt
        self.proportional = np.zeros(3)
        self.integral = np.zeros(3)
        self.derivative = np.zeros(3)
        self.last_error = None
        self.last_time = None

    def update(self, error_vector):
        error_vector = np.array(error_vector)
        current_time = time.time()

        self.proportional = error_vector

        if self.last_time is None:
            self.derivative = np.zeros(3)
        else:
            if self.last_error is not None:
                self.derivative = (error_vector - self.last_error) / self.dt
            else:
                self.derivative = np.zeros(3)

        self.integral += error_vector * self.dt
        self.integral = np.clip(self.integral, -1.0, 1.0)

        P_term = self.kp * self.proportional
        I_term = self.ki * self.integral
        D_term = self.kd * self.derivative

        output = P_term + I_term + D_term

        self.last_error = error_vector.copy()
        self.last_time = current_time

        return output, P_term, I_term, D_term

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
        self.lifting = False
        self.control_thread = None
        self.control_stop = threading.Event()

        self.control_rate_hz = hz
        self.force_pid = VectorPIDController(kp=kp_f, ki=ki_f, kd=kd_f, dt=1 / hz)
        self.pose_pid = VectorPIDController(kp=kp_p, ki=ki_p, kd=kd_p, dt=1 / hz)

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

    def control_to_target(
        self, reference_force=None, distance_cap=0.2, timeout=30.0, pose_updates=None
    ):
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

        self.pose_updates = pose_updates or []
        self.applied_updates = set()
        self.pose_updates.sort(key=lambda x: x["time"])

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

        self.lifting_applied = False
        self.side_applied = False
        self.down_applied = False

        while self.control_active and not self.control_stop.is_set():
            loop_start = time.time()

            try:
                current_state = self.get_state()

                current_force_vector = np.array(
                    current_state["filtered_force_world"][:3]
                )
                current_position = np.array(current_state["gripper_world"][:3])

                # Safety checks
                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break

                if time.time() - self.start_time >= self.control_timeout:
                    print(f"Control timeout reached")
                    break

                # Handle timed pose updates
                current_time = time.time() - self.start_time
                for i, update in enumerate(self.pose_updates):
                    if i in self.applied_updates:
                        continue

                    if current_time >= update["time"]:
                        offset = np.array(update["position"])
                        self.target_position = self.target_position + offset
                        self.applied_updates.add(i)

                # === 3D FORCE CONTROL ===
                if np.isscalar(self.ref_force):
                    ref_force_vector = self.ref_force * (
                        -self.control_direction
                    )  # opposite of control direction, same as the original normal
                else:
                    ref_force_vector = np.array(self.ref_force)
                force_error_vector = ref_force_vector - current_force_vector
                force_output_vector, force_p_term, force_i_term, force_d_term = (
                    self.force_pid.update(force_error_vector)
                )

                # === 3D POSITION CONTROL ===
                position_error_vector = self.target_position - current_position
                position_output_vector, pos_p_term, pos_i_term, pos_d_term = (
                    self.pose_pid.update(position_error_vector)
                )

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
                    "force_p_term": force_p_term.copy(),
                    "force_i_term": force_i_term.copy(),
                    "force_d_term": force_d_term.copy(),
                    "pos_p_term": pos_p_term.copy(),
                    "pos_i_term": pos_i_term.copy(),
                    "pos_d_term": pos_d_term.copy(),
                }
                self.control_data.append(data_point)

                self.speedL_world(speed_command, acceleration=0.1, time_duration=0.01)

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

    def plot_PID(self):
        """Generate comprehensive plots for logged PID components"""
        if not self.control_data:
            print("No data to plot")
            return

        # Check if PID components are logged
        if "force_p_term" not in self.control_data[0]:
            print(
                "PID components not logged in control data. Make sure you're using the updated _control_loop3D method."
            )
            return

        # Extract data from logged data
        timestamps = [d["timestamp"] for d in self.control_data]

        # Extract logged PID components
        force_p_terms = np.array([d["force_p_term"] for d in self.control_data])
        force_i_terms = np.array([d["force_i_term"] for d in self.control_data])
        force_d_terms = np.array([d["force_d_term"] for d in self.control_data])

        pos_p_terms = np.array([d["pos_p_term"] for d in self.control_data])
        pos_i_terms = np.array([d["pos_i_term"] for d in self.control_data])
        pos_d_terms = np.array([d["pos_d_term"] for d in self.control_data])

        # Calculate combined terms
        force_total = force_p_terms + force_i_terms + force_d_terms
        pos_total = pos_p_terms + pos_i_terms + pos_d_terms
        combined_total = force_total + pos_total

        # Setup plot
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

        plt.figure(figsize=(20, 12))

        # Main title
        plt.suptitle(
            f"PID Components Analysis ({current_datetime})\n"
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
            fontsize=16,
            y=0.98,
        )

        colors = ["red", "green", "blue"]
        axis_labels = ["X", "Y", "Z"]

        # === COLUMN 1: Force PID Components ===

        # 1. Force P Terms
        plt.subplot(3, 3, 1)
        for axis in range(3):
            plt.plot(
                timestamps,
                force_p_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Force PID - Proportional Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("P Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 4. Force I Terms
        plt.subplot(3, 3, 4)
        for axis in range(3):
            plt.plot(
                timestamps,
                force_i_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Force PID - Integral Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("I Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 7. Force D Terms
        plt.subplot(3, 3, 7)
        for axis in range(3):
            plt.plot(
                timestamps,
                force_d_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Force PID - Derivative Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("D Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # === COLUMN 2: Position PID Components ===

        # 2. Position P Terms
        plt.subplot(3, 3, 2)
        for axis in range(3):
            plt.plot(
                timestamps,
                pos_p_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Position PID - Proportional Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("P Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 5. Position I Terms
        plt.subplot(3, 3, 5)
        for axis in range(3):
            plt.plot(
                timestamps,
                pos_i_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Position PID - Integral Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("I Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 8. Position D Terms
        plt.subplot(3, 3, 8)
        for axis in range(3):
            plt.plot(
                timestamps,
                pos_d_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Position PID - Derivative Terms")
        plt.xlabel("Time (s)")
        plt.ylabel("D Term Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # === COLUMN 3: Combined P, I, D Terms ===

        # 3. Combined P Terms (Force + Position)
        plt.subplot(3, 3, 3)
        combined_p_terms = force_p_terms + pos_p_terms
        for axis in range(3):
            plt.plot(
                timestamps,
                combined_p_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Combined P Terms (Force + Position)")
        plt.xlabel("Time (s)")
        plt.ylabel("Combined P Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 6. Combined I Terms (Force + Position)
        plt.subplot(3, 3, 6)
        combined_i_terms = force_i_terms + pos_i_terms
        for axis in range(3):
            plt.plot(
                timestamps,
                combined_i_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Combined I Terms (Force + Position)")
        plt.xlabel("Time (s)")
        plt.ylabel("Combined I Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        # 9. Combined D Terms (Force + Position)
        plt.subplot(3, 3, 9)
        combined_d_terms = force_d_terms + pos_d_terms
        for axis in range(3):
            plt.plot(
                timestamps,
                combined_d_terms[:, axis],
                label=f"{axis_labels[axis]} Axis",
                linewidth=2,
                color=colors[axis],
            )
        plt.title("Combined D Terms (Force + Position)")
        plt.xlabel("Time (s)")
        plt.ylabel("Combined D Output")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot
        filename = (
            f"plots/pid_components_analysis_{current_datetime}_{self.robot_id}.png"
        )
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

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
    hz = 100
    reference_force = 120  # 150
    base_force = 12.5
    factor = base_force / reference_force

    kp_f = 0.0015 * factor
    ki_f = 0.0000 * factor
    kd_f = 0.000025 * factor

    kp_p = 1  # 0.5
    ki_p = 0.00005
    kd_p = 0.34  # 0.0025

    alpha = 0.99
    timeout = 20

    lifting = [
        {"time": 4.0, "position": [0, 0, 0.2]},
        {"time": 8.0, "position": [-1, 0, 0]},
        {"time": 16.0, "position": [0, 0, -0.2]},
    ]

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
        robotL.moveJ(
            [-2.72771532, -1.40769446, 2.81887228, -3.01955523, -1.6224683, 2.31350756]
        )

        robotL.wait_for_commands()

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
            reference_force=reference_force,
            distance_cap=1.5,
            timeout=timeout,
            pose_updates=lifting,
        )

        robotL.control_to_target(
            reference_force=reference_force,
            distance_cap=1.5,
            timeout=timeout,
            pose_updates=lifting,
        )

        robotR.wait_for_control()
        robotL.wait_for_control()

        robotR.plot_data3D()
        robotR.plot_PID()
        robotL.plot_data3D()
        robotL.plot_PID()

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
