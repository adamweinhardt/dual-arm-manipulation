import time
import threading
import numpy as np
import zmq
import os
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from control.ur_controller import URController
from scipy.spatial.transform import Rotation as R
from utils.utils import (
    rotmat_to_rvec,
    end_effector_rotation_from_normal,
    _assert_rotmat,
)


class VectorPIDController:
    """3D Vector PID Controller - separate PID for each axis"""

    def __init__(self, kp, ki, kd, dt=0.02):
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
            self.derivative = (
                (error_vector - self.last_error) / self.dt
                if self.last_error is not None
                else np.zeros(3)
            )

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
        self.integral = np.zeros(3)
        self.last_error = None
        self.last_time = None


class URForceController(URController):
    """URController with force control capabilities (debug-instrumented)"""

    def __init__(
        self,
        ip,
        hz=50,
        kp_f=0.01,
        ki_f=0,
        kd_f=0,
        kp_p=0.01,
        ki_p=0,
        kd_p=0.0,
        kp_r=0,
        ki_r=0,
        kd_r=0,
    ):
        super().__init__(ip)

        self.control_active = False
        self.lifting = False
        self.control_thread = None
        self.control_stop = threading.Event()

        self.control_rate_hz = hz
        self.force_pid = VectorPIDController(kp=kp_f, ki=ki_f, kd=kd_f, dt=1 / hz)
        self.pose_pid = VectorPIDController(kp=kp_p, ki=ki_p, kd=kd_p, dt=1 / hz)
        self.rot_pid = VectorPIDController(kp=kp_r, ki=ki_r, kd=kd_r, dt=1 / hz)

        self.control_data = []

        # ZMQ
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
            return False
        except zmq.Again:
            return False
        except Exception as e:
            print(f"Error receiving grasping data: {e}")
            return False

    def get_grasping_data(self):
        """Get approach point for this robot from latest grasping data"""
        while not self._update_grasping_data():
            time.sleep(0.01)

        box_id = list(self.current_grasping_data.keys())[0]
        g = self.current_grasping_data[box_id]

        if self.robot_id == 0:
            grasping_point = g.get("grasping_point0")
            approach_point = g.get("approach_point0")
            normal_vector = g.get("normal0")
        elif self.robot_id == 1:
            grasping_point = g.get("grasping_point1")
            approach_point = g.get("approach_point1")
            normal_vector = g.get("normal1")
        else:
            return None, None, None

        if approach_point is None or normal_vector is None:
            return None, None, None

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
        reference_position=None,
        reference_rotation=None,
        reference_force=None,
        direction=[0, 0, -1],
        distance_cap=0.2,
        timeout=30.0,
        pose_updates=None,
    ):
        if self.control_active:
            print("Control already active!")
            return False

        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        state = self.get_state()
        self.start_position = np.array(state["gripper_world"][:3])
        self.start_rotation = np.array(state["pose"][3:6])  # rotvec

        # references
        self.reference_position = (
            np.array(reference_position)
            if reference_position is not None
            else self.start_position.copy()
        )
        self.reference_rotation = (
            np.array(reference_rotation)
            if reference_rotation is not None
            else self.start_rotation.copy()
        )
        self.reference_rotation_matrix = _assert_rotmat(
            "my ref_R (manual init)", R.from_rotvec(self.reference_rotation).as_matrix()
        )

        self.ref_force = 0.0 if reference_force is None else reference_force
        self.control_direction = direction
        self.distance_cap = distance_cap
        self.control_timeout = timeout
        self.start_time = time.time()

        self.pose_updates = pose_updates or []
        self.applied_updates = set()
        self.pose_updates.sort(key=lambda x: x["time"])

        # reset
        self.force_pid.reset()
        self.pose_pid.reset()
        self.rot_pid.reset()
        self.control_data = []
        self.trajectory = None
        self.deadzone_threshold = None

        self.rtde_control.zeroFtSensor()

        self.control_active = True
        self.control_stop.clear()
        self.control_thread = threading.Thread(target=self._control_loop3D, daemon=True)
        self.control_thread.start()

    # ---------------- shared frame helpers ----------------
    def _compute_initial_box_frame(self, pA, pB):
        x_hat = (pB - pA) / (np.linalg.norm(pB - pA) + 1e-9)
        z_hat = np.array([0, 0, 1]) - np.dot([0, 0, 1], x_hat) * x_hat
        if np.linalg.norm(z_hat) < 1e-6:
            z_hat = np.array([0, 1, 0]) - np.dot([0, 1, 0], x_hat) * x_hat
        z_hat /= np.linalg.norm(z_hat) + 1e-9
        y_hat = np.cross(z_hat, x_hat)
        return np.column_stack([x_hat, y_hat, z_hat]), 0.5 * (pA + pB)

    def _init_box_frame_and_grasp(self, grasping_point, grasping_point_other, R_WG0):
        self._R_WB0, self._p_WB0 = self._compute_initial_box_frame(
            grasping_point, grasping_point_other
        )
        self._R_BG = self._R_WB0.T @ R_WG0
        self._r_B = self._R_WB0.T @ (grasping_point - self._p_WB0)

    def _compute_reference_rotation(self, R_B0B):
        R_WB = self._R_WB0 @ R_B0B
        return R_WB @ self._R_BG

    def control_to_target(
        self,
        reference_force=0.0,
        distance_cap=0.2,
        timeout=30.0,
        trajectory=None,
        deadzone_threshold=None,
    ):
        if self.control_active:
            print("Control already active!")
            return False

        # Grasp and init
        grasping_point, _, normal = self.get_grasping_data()
        state = self.get_state()
        self.start_position = np.array(state["gripper_world"][:3])
        self.start_rotation = np.array(state["pose"][3:6])  # rotvec
        self.grasping_point = grasping_point

        # defaults
        self.reference_position = np.array(grasping_point, dtype=float)
        self.reference_rotation = self.start_rotation.copy()
        self.reference_rotation_matrix = _assert_rotmat(
            "my ref_R (init)", R.from_rotvec(self.reference_rotation).as_matrix()
        )

        self.ref_force = float(reference_force)
        self.control_direction = -np.array(normal, dtype=float) / (
            np.linalg.norm(normal) + 1e-9
        )
        self.distance_cap = distance_cap
        self.control_timeout = timeout
        self.deadzone_threshold = deadzone_threshold
        self.start_time = time.time()

        if trajectory:
            traj_npz = np.load(trajectory)
            self.rot_updates = traj_npz["rotation_matrices"]
            self.pose_updates_stream = traj_npz.get("position", None)

            ru = np.asarray(self.rot_updates)
            if ru.ndim == 2 and ru.shape[1] == 3:
                self.rot_updates = np.stack(
                    [R.from_rotvec(v).as_matrix() for v in ru], axis=0
                )
            elif ru.ndim == 3 and ru.shape[1:] == (3, 3):
                pass
            else:
                raise ValueError(f"rotation_matrices has unexpected shape {ru.shape}")
            self._traj_len = len(self.rot_updates)

            if (
                not hasattr(self, "other_robot_grasp_point")
                or self.other_robot_grasp_point is None
            ):
                pass
            self._init_box_frame_and_grasp(
                np.array(grasping_point),
                np.array(self.other_robot_grasp_point),
                R.from_rotvec(self.start_rotation).as_matrix(),
            )
        else:
            self.rot_updates = None
            self.pose_updates_stream = None
            self._traj_len = 0

        # Reset PIDs & logs
        self.force_pid.reset()
        self.pose_pid.reset()
        self.rot_pid.reset()
        self.control_data.clear()

        self.rtde_control.zeroFtSensor()
        self.control_active = True
        self.control_stop.clear()
        self.control_thread = threading.Thread(target=self._control_loop3D, daemon=True)
        self.control_thread.start()

    # ---------------------------
    # Control Loop
    # ---------------------------
    def _control_loop3D(self):
        control_period = 1.0 / self.control_rate_hz

        trajectory_started = False
        trajectory_index = 0

        while self.control_active and not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                state = self.get_state()
                current_force_vector = np.array(state["filtered_force_world"][:3])
                current_position = np.array(state["gripper_world"][:3])
                current_rotation = np.array(state["pose"][3:6])  # rotvec
                current_rotation_matrix = R.from_rotvec(current_rotation).as_matrix()

                # --- Safety ---
                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break
                if time.time() - self.start_time >= self.control_timeout:
                    print("Control timeout reached")
                    break

                current_time = time.time() - self.start_time

                # Start trajectory after 3s
                if current_time >= 3.0 and not trajectory_started:
                    trajectory_started = True
                    print(f"Starting trajectory at t={current_time:.1f}s")

                # Trajectory
                if (
                    self.rot_updates is not None
                    and trajectory_started
                    and trajectory_index < self._traj_len
                ):
                    position_offset = (
                        self.pose_updates_stream[trajectory_index]
                        if self.pose_updates_stream is not None
                        else np.zeros(3)
                    )
                    self.reference_position = self.grasping_point + position_offset

                    R_B0B = _assert_rotmat(
                        "R_B0B(my traj)", self.rot_updates[trajectory_index]
                    )
                    R_WG_ref = self._compute_reference_rotation(R_B0B)
                    self.reference_rotation_matrix = _assert_rotmat(
                        "my ref_R (traj)", R_WG_ref
                    )
                    self.reference_rotation = R.from_matrix(
                        self.reference_rotation_matrix
                    ).as_rotvec()

                    trajectory_index += 1

                # --- Force PID ---
                ref_force_vector = (
                    self.ref_force * (-self.control_direction)
                    if np.isscalar(self.ref_force)
                    else np.array(self.ref_force)
                )
                force_error_vector = ref_force_vector - current_force_vector
                force_output_vector, force_p_term, force_i_term, force_d_term = (
                    self.force_pid.update(force_error_vector)
                )

                # --- Position PID ---
                position_error_vector = self.reference_position - current_position
                position_output_vector, pos_p_term, pos_i_term, pos_d_term = (
                    self.pose_pid.update(position_error_vector)
                )
                if self.deadzone_threshold is not None:
                    for i in range(3):
                        if abs(position_error_vector[i]) <= self.deadzone_threshold:
                            position_output_vector[i] = 0.0
                            pos_p_term[i] = pos_i_term[i] = pos_d_term[i] = 0.0

                total_output_vector = -force_output_vector + position_output_vector

                # --- Rotation PID ---
                self.reference_rotation_matrix = _assert_rotmat(
                    "my ref_R (pre-err)", self.reference_rotation_matrix
                )
                error_matrix = (
                    self.reference_rotation_matrix @ current_rotation_matrix.T
                )
                rotation_error_vector = rotmat_to_rvec(error_matrix)
                rotation_output_vector, rot_p_term, rot_i_term, rot_d_term = (
                    self.rot_pid.update(rotation_error_vector)
                )

                # --- Command ---
                speed_command = [*total_output_vector, *rotation_output_vector]
                self.speedL_world(speed_command, acceleration=0.18, time_duration=0.001)

                # --- Log ---
                self.control_data.append(
                    {
                        "timestamp": time.time() - self.start_time,
                        "force_vector": current_force_vector,
                        "position": current_position,
                        "rotation": current_rotation,
                        "reference_position": self.reference_position.copy(),
                        "reference_rotation": self.reference_rotation.copy(),
                        "reference_force_vector": ref_force_vector,
                        "position_error_vector": position_error_vector,
                        "rotation_error_vector": rotation_error_vector,
                        "force_error_vector": force_error_vector,
                        "position_output_vector": position_output_vector,
                        "rotation_output_vector": rotation_output_vector,
                        "force_output_vector": force_output_vector,
                        "total_output_vector": total_output_vector,
                        "force_p_term": force_p_term,
                        "force_i_term": force_i_term,
                        "force_d_term": force_d_term,
                        "pos_p_term": pos_p_term,
                        "pos_i_term": pos_i_term,
                        "pos_d_term": pos_d_term,
                        "rot_p_term": rot_p_term,
                        "rot_i_term": rot_i_term,
                        "rot_d_term": rot_d_term,
                        "deadzone_threshold": self.deadzone_threshold,
                    }
                )

            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.speedStop()
                except:
                    pass
                break

            # Keep loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, control_period - elapsed))

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
        super().disconnect

    def plot_PID(self):
        """Generate comprehensive plots for logged PID components including rotation"""
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

        # Check if rotation PID data exists
        has_rotation_data = "rot_p_term" in self.control_data[0]
        if has_rotation_data:
            rot_p_terms = np.array([d["rot_p_term"] for d in self.control_data])
            rot_i_terms = np.array([d["rot_i_term"] for d in self.control_data])
            rot_d_terms = np.array([d["rot_d_term"] for d in self.control_data])

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

        # Get rotation PID string if available
        rot_pid_str = ""
        if has_rotation_data:
            if hasattr(self, "rot_pid"):
                pid_obj = self.rot_pid
            elif hasattr(self, "rot_pid"):
                pid_obj = self.rot_pid
            else:
                pid_obj = None

            if pid_obj:
                if hasattr(pid_obj, "kp") and np.isscalar(pid_obj.kp):
                    rot_pid_str = (
                        f"Kp={pid_obj.kp:.3f}, Ki={pid_obj.ki:.3f}, Kd={pid_obj.kd:.3f}"
                    )
                else:
                    rot_pid_str = f"Kp={pid_obj.kp}, Ki={pid_obj.ki}, Kd={pid_obj.kd}"

        # Create figure with appropriate size
        fig_width = 30 if has_rotation_data else 20
        plt.figure(figsize=(fig_width, 12))

        # Main title
        title_parts = [
            f"PID Components Analysis ({current_datetime})",
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
        ]
        if has_rotation_data:
            title_parts.append(f"Rotation PID: ({rot_pid_str})")

        plt.suptitle("\n".join(title_parts), fontsize=16, y=0.98)

        colors = ["red", "green", "blue"]
        axis_labels = ["X", "Y", "Z"]
        n_cols = 4 if has_rotation_data else 3

        # === COLUMN 1: Force PID Components ===

        # 1. Force P Terms
        plt.subplot(3, n_cols, 1)
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
        plt.subplot(3, n_cols, n_cols + 1)
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
        plt.subplot(3, n_cols, 2 * n_cols + 1)
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
        plt.subplot(3, n_cols, 2)
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
        plt.subplot(3, n_cols, n_cols + 2)
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
        plt.subplot(3, n_cols, 2 * n_cols + 2)
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

        # === COLUMN 3: Rotation PID Components (if available) ===
        if has_rotation_data:
            rot_axis_labels = ["Rx", "Ry", "Rz"]

            # 3. Rotation P Terms
            plt.subplot(3, n_cols, 3)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    rot_p_terms[:, axis],
                    label=f"{rot_axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Rotation PID - Proportional Terms")
            plt.xlabel("Time (s)")
            plt.ylabel("P Term Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # 6. Rotation I Terms
            plt.subplot(3, n_cols, n_cols + 3)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    rot_i_terms[:, axis],
                    label=f"{rot_axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Rotation PID - Integral Terms")
            plt.xlabel("Time (s)")
            plt.ylabel("I Term Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # 9. Rotation D Terms
            plt.subplot(3, n_cols, 2 * n_cols + 3)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    rot_d_terms[:, axis],
                    label=f"{rot_axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Rotation PID - Derivative Terms")
            plt.xlabel("Time (s)")
            plt.ylabel("D Term Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # === COLUMN 4: Combined Terms ===
            # Calculate combined terms including rotation
            combined_p_terms = force_p_terms + pos_p_terms + rot_p_terms
            combined_i_terms = force_i_terms + pos_i_terms + rot_i_terms
            combined_d_terms = force_d_terms + pos_d_terms + rot_d_terms

            # 4. Combined P Terms
            plt.subplot(3, n_cols, 4)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    combined_p_terms[:, axis],
                    label=f"{axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Combined P Terms (Force + Position + Rotation)")
            plt.xlabel("Time (s)")
            plt.ylabel("Combined P Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # 7. Combined I Terms
            plt.subplot(3, n_cols, n_cols + 4)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    combined_i_terms[:, axis],
                    label=f"{axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Combined I Terms (Force + Position + Rotation)")
            plt.xlabel("Time (s)")
            plt.ylabel("Combined I Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # 10. Combined D Terms
            plt.subplot(3, n_cols, 2 * n_cols + 4)
            for axis in range(3):
                plt.plot(
                    timestamps,
                    combined_d_terms[:, axis],
                    label=f"{axis_labels[axis]} Axis",
                    linewidth=2,
                    color=colors[axis],
                )
            plt.title("Combined D Terms (Force + Position + Rotation)")
            plt.xlabel("Time (s)")
            plt.ylabel("Combined D Output")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        else:
            # === COLUMN 3: Combined Terms (without rotation) ===
            combined_p_terms = force_p_terms + pos_p_terms
            combined_i_terms = force_i_terms + pos_i_terms
            combined_d_terms = force_d_terms + pos_d_terms

            # 3. Combined P Terms
            plt.subplot(3, n_cols, 3)
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

            # 6. Combined I Terms
            plt.subplot(3, n_cols, n_cols + 3)
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

            # 9. Combined D Terms
            plt.subplot(3, n_cols, 2 * n_cols + 3)
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
        plot_type = "6dof" if has_rotation_data else "3dof"
        filename = f"plots/pid_components_analysis_{plot_type}_{current_datetime}_{self.robot_id}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"PID plot saved: {filename}")

    def plot_data3D(self):
        """Generate comprehensive plots for 6DOF control data including rotation tracking"""
        if not self.control_data:
            print("No data to plot")
            return

        # Extract data from logged data
        timestamps = [d["timestamp"] for d in self.control_data]
        force_vectors = np.array([d["force_vector"] for d in self.control_data])
        positions = np.array([d["position"] for d in self.control_data])
        reference_positions = np.array(
            [d["reference_position"] for d in self.control_data]
        )

        # Check if rotation data exists
        has_rotation_data = (
            "rotation" in self.control_data[0]
            and "reference_rotation" in self.control_data[0]
        )

        if has_rotation_data:
            rotations = np.array([d["rotation"] for d in self.control_data])
            reference_rotations = np.array(
                [d["reference_rotation"] for d in self.control_data]
            )

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

        if has_rotation_data:
            rotation_error_vectors = np.array(
                [d["rotation_error_vector"] for d in self.control_data]
            )
            rotation_output_vectors = np.array(
                [d["rotation_output_vector"] for d in self.control_data]
            )

        # Handle both 3D and 6D reference positions (extract only position part)
        if reference_positions.shape[1] == 6:
            reference_positions = reference_positions[:, :3]
        elif reference_positions.shape[1] != 3:
            print(
                f"Warning: Unexpected reference_position shape {reference_positions.shape}, using first 3 elements"
            )
            reference_positions = reference_positions[:, :3]

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

        if has_rotation_data:
            rotation_error_magnitudes = np.linalg.norm(rotation_error_vectors, axis=1)
            rotation_output_magnitudes = np.linalg.norm(rotation_output_vectors, axis=1)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        initial_target_pose_str = f"[{self.reference_position[0]:.3f}, {self.reference_position[1]:.3f}, {self.reference_position[2]:.3f}]"

        # PID parameters
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

        # Create figure with appropriate size
        fig_width = 24 if has_rotation_data else 18
        plt.figure(figsize=(fig_width, 12))

        n_cols = 4 if has_rotation_data else 3

        # Main title for the entire figure
        ref_force_str = (
            f"{reference_force_magnitudes[0]:.2f}N"
            if len(reference_force_magnitudes) > 0
            else "N/A"
        )
        control_type = "6DOF" if has_rotation_data else "3DOF"

        title_parts = [
            f"{control_type} Vector Control Data Plot ({current_datetime})",
            f"Ref Force: {ref_force_str}, Target Pose: {initial_target_pose_str}",
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
        ]

        plt.suptitle("\n".join(title_parts), fontsize=14, y=0.98)

        colors = ["red", "green", "blue"]
        axis_labels = ["X", "Y", "Z"]
        rot_axis_labels = ["Rx", "Ry", "Rz"]

        # === ROW 1: Reference Tracking and Movement ===

        # 1. Force vs Target Force (3D Components)
        plt.subplot(3, n_cols, 1)
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
        plt.subplot(3, n_cols, 2)
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
            reference_positions[:, 0],
            label="X Target",
            linewidth=2,
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            reference_positions[:, 1],
            label="Y Target",
            linewidth=2,
            linestyle="--",
            color="green",
            alpha=0.7,
        )
        plt.plot(
            timestamps,
            reference_positions[:, 2],
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

        # 3. Rotation vs Target Rotation (if available)
        if has_rotation_data:
            plt.subplot(3, n_cols, 3)
            plt.plot(
                timestamps,
                rotations[:, 0],
                label="Rx Current",
                linewidth=2,
                color="red",
            )
            plt.plot(
                timestamps,
                rotations[:, 1],
                label="Ry Current",
                linewidth=2,
                color="green",
            )
            plt.plot(
                timestamps,
                rotations[:, 2],
                label="Rz Current",
                linewidth=2,
                color="blue",
            )
            plt.plot(
                timestamps,
                reference_rotations[:, 0],
                label="Rx Target",
                linewidth=2,
                linestyle="--",
                color="red",
                alpha=0.7,
            )
            plt.plot(
                timestamps,
                reference_rotations[:, 1],
                label="Ry Target",
                linewidth=2,
                linestyle="--",
                color="green",
                alpha=0.7,
            )
            plt.plot(
                timestamps,
                reference_rotations[:, 2],
                label="Rz Target",
                linewidth=2,
                linestyle="--",
                color="blue",
                alpha=0.7,
            )
            plt.title("Rotation vs Target Rotation")
            plt.xlabel("Time (s)")
            plt.ylabel("Rotation (rad)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 4. Movement Distance vs Time
            plt.subplot(3, n_cols, 4)
        else:
            # 3. Movement Distance vs Time
            plt.subplot(3, n_cols, 3)

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

        # Force Error Vectors (3D)
        plt.subplot(3, n_cols, n_cols + 1)
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

        # Position Error Vectors (3D)
        plt.subplot(3, n_cols, n_cols + 2)
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
        deadzone_threshold = self.control_data[0].get("deadzone_threshold", None)
        if deadzone_threshold is not None:
            plt.axhline(
                y=deadzone_threshold,
                color="orange",
                linestyle="dotted",
                linewidth=2,
                label=f"Deadzone +{deadzone_threshold:.2f}m",
            )
            plt.axhline(
                y=-deadzone_threshold,
                color="orange",
                linestyle="dotted",
                linewidth=2,
                label=f"Deadzone -{deadzone_threshold:.2f}m",
            )
        plt.title("3D Position Error Vectors")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rotation Error Vectors (if available)
        if has_rotation_data:
            plt.subplot(3, n_cols, n_cols + 3)
            plt.plot(
                timestamps,
                rotation_error_vectors[:, 0],
                label="Rotation Error Rx",
                linewidth=2,
                color="red",
            )
            plt.plot(
                timestamps,
                rotation_error_vectors[:, 1],
                label="Rotation Error Ry",
                linewidth=2,
                color="green",
            )
            plt.plot(
                timestamps,
                rotation_error_vectors[:, 2],
                label="Rotation Error Rz",
                linewidth=2,
                color="blue",
            )
            plt.plot(
                timestamps,
                rotation_error_magnitudes,
                label="Rotation Error Magnitude",
                linewidth=2,
                color="black",
                linestyle="--",
            )
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            plt.title("3D Rotation Error Vectors")
            plt.xlabel("Time (s)")
            plt.ylabel("Rotation Error (rad)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Directional Errors
            plt.subplot(3, n_cols, n_cols + 4)
        else:
            # Directional Errors
            plt.subplot(3, n_cols, n_cols + 3)

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

        # Force Output Vectors (3D)
        plt.subplot(3, n_cols, 2 * n_cols + 1)
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

        # Position Output Vectors (3D)
        plt.subplot(3, n_cols, 2 * n_cols + 2)
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

        # Rotation Output Vectors (if available)
        if has_rotation_data:
            plt.subplot(3, n_cols, 2 * n_cols + 3)
            plt.plot(
                timestamps,
                rotation_output_vectors[:, 0],
                label="Rotation Output Rx",
                linewidth=2,
                color="red",
            )
            plt.plot(
                timestamps,
                rotation_output_vectors[:, 1],
                label="Rotation Output Ry",
                linewidth=2,
                color="green",
            )
            plt.plot(
                timestamps,
                rotation_output_vectors[:, 2],
                label="Rotation Output Rz",
                linewidth=2,
                color="blue",
            )
            plt.plot(
                timestamps,
                rotation_output_magnitudes,
                label="Rotation Output Magnitude",
                linewidth=2,
                color="black",
                linestyle="--",
            )
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            plt.title("3D Rotation Control Outputs")
            plt.xlabel("Time (s)")
            plt.ylabel("Rotation Output (rad/s)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Total Output Vectors (3D)
            plt.subplot(3, n_cols, 2 * n_cols + 4)
        else:
            # Total Output Vectors (3D)
            plt.subplot(3, n_cols, 2 * n_cols + 3)

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
        plt.title("3D Total Linear Outputs")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Output")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save filename with date
        plot_type = "6dof" if has_rotation_data else "3dof"
        filename = f"plots/comprehensive_{plot_type}_control_plot_{current_datetime}_{self.robot_id}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        # # Print comprehensive summary statistics
        # if timestamps:
        #     final_force_mag = force_magnitudes[-1] if len(force_magnitudes) > 0 else 0
        #     final_ref_force_mag = (
        #         reference_force_magnitudes[-1]
        #         if len(reference_force_magnitudes) > 0
        #         else 0
        #     )
        #     final_force_in_dir = force_in_direction[-1] if force_in_direction else 0
        #     final_ref_force_in_dir = (
        #         ref_force_in_direction[-1] if ref_force_in_direction else 0
        #     )

        #     max_distance = (
        #         max(distances_from_start) if len(distances_from_start) > 0 else 0
        #     )
        #     control_duration = timestamps[-1]

        #     final_position_error_mag = (
        #         position_error_magnitudes[-1]
        #         if len(position_error_magnitudes) > 0
        #         else 0
        #     )
        #     final_position_error_dir = (
        #         position_errors_in_direction[-1] if position_errors_in_direction else 0
        #     )

        #     avg_force_error_mag = np.mean(force_error_magnitudes)
        #     avg_position_error_mag = np.mean(position_error_magnitudes)
        #     avg_force_error_dir = np.mean(np.abs(force_errors_in_direction))
        #     avg_position_error_dir = np.mean(np.abs(position_errors_in_direction))

        #     control_summary_title = "6DOF" if has_rotation_data else "3DOF"
        #     print("=" * 70)
        #     print(f"COMPREHENSIVE {control_summary_title} VECTOR CONTROL SUMMARY")
        #     print("=" * 70)
        #     print(f"Plot saved as: {filename} ({len(self.control_data)} samples)")
        #     print(f"Control duration: {control_duration:.2f}s")
        #     print("")
        #     print("FORCE TRACKING:")
        #     print(f"  Final force magnitude: {final_force_mag:.3f}N")
        #     print(f"  Reference force magnitude: {final_ref_force_mag:.3f}N")
        #     print(f"  Final force in direction: {final_force_in_dir:.3f}N")
        #     print(f"  Reference force in direction: {final_ref_force_in_dir:.3f}N")
        #     print(
        #         f"  Force error magnitude: {abs(final_ref_force_mag - final_force_mag):.3f}N"
        #     )
        #     print(f"  Average force error magnitude: {avg_force_error_mag:.3f}N")
        #     print(f"  Average force error in direction: {avg_force_error_dir:.3f}N")
        #     print("")
        #     print("POSITION TRACKING:")
        #     print(f"  Max distance moved: {max_distance:.3f}m")
        #     if hasattr(self, "distance_cap"):
        #         print(f"  Distance cap: {self.distance_cap}m")
        #     print(f"  Final position error magnitude: {final_position_error_mag:.3f}m")
        #     print(
        #         f"  Final position error in direction: {final_position_error_dir:.3f}m"
        #     )
        #     print(f"  Average position error magnitude: {avg_position_error_mag:.3f}m")
        #     print(
        #         f"  Average position error in direction: {avg_position_error_dir:.3f}m"
        #     )

        #     # Add rotation tracking summary if available
        #     if has_rotation_data:
        #         final_rotation_error_mag = (
        #             rotation_error_magnitudes[-1]
        #             if len(rotation_error_magnitudes) > 0
        #             else 0
        #         )
        #         avg_rotation_error_mag = np.mean(rotation_error_magnitudes)

        #         print("")
        #         print("ROTATION TRACKING:")
        #         print(
        #             f"  Final rotation error magnitude: {final_rotation_error_mag:.3f}°"
        #         )
        #         print(
        #             f"  Average rotation error magnitude: {avg_rotation_error_mag:.3f}°"
        #         )

        #         final_rotation_deg = rotations[-1] if len(rotations) > 0 else [0, 0, 0]
        #         final_ref_rotation_deg = (
        #             reference_rotations[-1]
        #             if len(reference_rotations) > 0
        #             else [0, 0, 0]
        #         )
        #         print(
        #             f"  Final rotation: [{final_rotation_deg[0]:.2f}°, {final_rotation_deg[1]:.2f}°, {final_rotation_deg[2]:.2f}°]"
        #         )
        #         print(
        #             f"  Target rotation: [{final_ref_rotation_deg[0]:.2f}°, {final_ref_rotation_deg[1]:.2f}°, {final_ref_rotation_deg[2]:.2f}°]"
        #         )

        #     print("")
        #     print("CONTROL PARAMETERS:")
        #     print(
        #         f"  Control direction: [{self.control_direction[0]:.2f}, {self.control_direction[1]:.2f}, {self.control_direction[2]:.2f}]"
        #     )
        #     print(f"  Force PID: {force_pid_str}")
        #     print(f"  Pose PID: {pose_pid_str}")

        #     if has_rotation_data:
        #         # Get rotation PID string
        #         if hasattr(self, "rot_pid"):
        #             pid_obj = self.rot_pid
        #         elif hasattr(self, "rot_pid"):
        #             pid_obj = self.rot_pid
        #         else:
        #             pid_obj = None

        #         if pid_obj:
        #             if hasattr(pid_obj, "kp") and np.isscalar(pid_obj.kp):
        #                 rot_pid_str = f"Kp={pid_obj.kp:.3f}, Ki={pid_obj.ki:.3f}, Kd={pid_obj.kd:.3f}"
        #             else:
        #                 rot_pid_str = (
        #                     f"Kp={pid_obj.kp}, Ki={pid_obj.ki}, Kd={pid_obj.kd}"
        #                 )
        #             print(f"  Rotation PID: {rot_pid_str}")

        #     print("")
        #     print("FINAL OUTPUT MAGNITUDES:")
        #     print(f"  Force output: {force_output_magnitudes[-1]:.6f}")
        #     print(f"  Position output: {position_output_magnitudes[-1]:.6f}")
        #     if has_rotation_data:
        #         print(f"  Rotation output: {rotation_output_magnitudes[-1]:.6f}")
        #     print(f"  Total linear output: {total_output_magnitudes[-1]:.6f}")
        #     print("=" * 70)
        # else:
        #     print("No data available for summary statistics")
