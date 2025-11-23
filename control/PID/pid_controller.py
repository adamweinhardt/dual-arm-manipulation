import time
import threading
import numpy as np
import zmq
import os
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from control.UR.ur_controller import URController
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
        kp_t=0,
        ki_t=0, 
        kd_t=0
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
        self.torque_pid = VectorPIDController(kp=kp_t, ki=ki_t, kd=kd_t, dt=1 / hz)

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
    def _canonicalize_pair(self, pA, pB):
        d = pB - pA
        axis = int(np.argmax(np.abs(d)))  # 0=x, 1=y, 2=z (dominant separation)
        if d[axis] < 0:  # enforce positive along dominant axis
            pA, pB = pB, pA
        return pA, pB

    def _compute_initial_box_frame(self, pA, pB):
        # --- canonicalize ordering so both arms build the SAME frame ---
        pA, pB = self._canonicalize_pair(pA, pB)

        x_hat = (pB - pA) / (np.linalg.norm(pB - pA) + 1e-9)

        # project world-z onto plane orthogonal to x_hat; fallback if degenerate
        z_guess = np.array([0, 0, 1])
        z_hat = z_guess - np.dot(z_guess, x_hat) * x_hat
        if np.linalg.norm(z_hat) < 1e-6:
            z_guess = np.array([0, 1, 0])
            z_hat = z_guess - np.dot(z_guess, x_hat) * x_hat
        z_hat /= np.linalg.norm(z_hat) + 1e-9

        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat) + 1e-9  # (tiny numeric guard)

        R_WB0 = np.column_stack([x_hat, y_hat, z_hat])
        p_WB0 = 0.5 * (pA + pB)
        return R_WB0, p_WB0

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
            print(self._r_B)
        else:
            self.rot_updates = None
            self.pose_updates_stream = None
            self._traj_len = 0

        # Reset PIDs & logs
        self.force_pid.reset()
        self.pose_pid.reset()
        self.rot_pid.reset()
        self.torque_pid.reset()
        self.control_data.clear()

        self.rtde_control.zeroFtSensor()
        self.control_active = True
        self.control_stop.clear()
        self.control_thread = threading.Thread(target=self._control_loop3D, daemon=True)
        self.control_thread.start()

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
                current_torque_vector = np.array(state["torque_world"][:3])

                # ========================= Safety ===========================
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

                # =========================== Trajectory Updates ===========================
                if (
                    self.rot_updates is not None
                    and trajectory_started
                    and trajectory_index < self._traj_len
                ):
                    # Position
                    position_offset = (
                        self.pose_updates_stream[trajectory_index]
                        if self.pose_updates_stream is not None
                        else np.zeros(3)
                    )
                    R_B0B = _assert_rotmat(
                        "R_B0B(my traj)", self.rot_updates[trajectory_index]
                    )
                    if self.robot_id == 0:
                        offset = [-0.055, 0, 0]
                    else:
                        offset = [0.055, 0, 0]

                    delta_p_rot_B = (R_B0B - np.eye(3)) @ (self._r_B + offset)
                    delta_p_rot_W = self._R_WB0 @ delta_p_rot_B
                    self.reference_position = (
                        self.grasping_point + position_offset + delta_p_rot_W
                    )
                    # Rotation
                    R_WG_ref = self._compute_reference_rotation(R_B0B)
                    self.reference_rotation_matrix = _assert_rotmat(
                        "my ref_R (traj)", R_WG_ref
                    )
                    self.reference_rotation = R.from_matrix(
                        self.reference_rotation_matrix
                    ).as_rotvec()

                    # Force
                    R_WB = self._R_WB0 @ R_B0B
                    if not hasattr(self, "_rhat_B"):
                        self._rhat_B = self._r_B / (np.linalg.norm(self._r_B) + 1e-9)

                    outward_normal_world = (
                        R_WB @ self._rhat_B
                    )  # points out of the box face
                    self.control_direction = -outward_normal_world

                    trajectory_index += 1

                # =========================== PID Control ===========================
                # --------------------------- Linear Control ---------------------------
                # Force
                ref_force_vector = (
                    self.ref_force * (-self.control_direction)
                    if np.isscalar(self.ref_force)
                    else np.array(self.ref_force)
                )
                force_error_vector = ref_force_vector - current_force_vector
                force_output_vector, force_p_term, force_i_term, force_d_term = (
                    self.force_pid.update(force_error_vector)
                )

                # Position
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

                # --------------------------- Orientation Control ---------------------------

                # Torque
                torque_ref_vector = np.zeros(3)  # always zero
                torque_error_vector = torque_ref_vector - current_torque_vector
                torque_output_vector, torque_p_term, torque_i_term, torque_d_term = self.torque_pid.update(torque_error_vector)

                # Rotation
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

                orientation_output_vector = rotation_output_vector + torque_output_vector

                # Command
                speed_command = [*total_output_vector, *orientation_output_vector]
                self.speedL_world(speed_command, acceleration=0.18, time_duration=0.001)

                # --- Log ---
                self.control_data.append(
                    {
                        # --- meta/time ---
                        "timestamp": time.time() - self.start_time,

                        # --- linear: force ---
                        "force_vector": current_force_vector,                  # measured (N)
                        "reference_force_vector": ref_force_vector,            # target (N)
                        "force_error_vector": force_error_vector,              # err (N)
                        "force_output_vector": force_output_vector,            # PID out (m/s contrib)
                        "force_p_term": force_p_term, "force_i_term": force_i_term, "force_d_term": force_d_term,

                        # --- linear: position ---
                        "position": current_position,                          # measured (m)
                        "reference_position": self.reference_position.copy(),  # target (m)
                        "position_error_vector": position_error_vector,        # err (m)
                        "position_output_vector": position_output_vector,      # PID out (m/s contrib)
                        "pos_p_term": pos_p_term, "pos_i_term": pos_i_term, "pos_d_term": pos_d_term,

                        # --- orientation: rotation (pose tracking) ---
                        "rotation": current_rotation,                          # measured rotvec (rad)
                        "reference_rotation": self.reference_rotation.copy(),  # target rotvec (rad)
                        "rotation_error_vector": rotation_error_vector,        # err (rad)
                        "rotation_output_vector": rotation_output_vector,      # PID out (rad/s contrib)
                        "rot_p_term": rot_p_term, "rot_i_term": rot_i_term, "rot_d_term": rot_d_term,

                        # --- orientation: torque (wrench rejection) ---
                        "torque_vector": current_torque_vector,                # measured (Nm)
                        "torque_ref_vector": torque_ref_vector,                # == 0
                        "torque_error_vector": torque_error_vector,            # err (Nm)
                        "torque_output_vector": torque_output_vector,          # PID out (rad/s contrib)
                        "torque_p_term": torque_p_term, "torque_i_term": torque_i_term, "torque_d_term": torque_d_term,

                        # --- totals & misc ---
                        "total_output_vector": total_output_vector,            # linear m/s = -force + position
                        "orientation_output_vector": orientation_output_vector,# rad/s = rotation + torque
                        "deadzone_threshold": self.deadzone_threshold,
                        "speed_command": np.array(speed_command),              # [vx vy vz wx wy wz]
                        "start_position": self.start_position.copy(),          # for distance plot
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
        """Five-column dashboard:
        Force | Position | Rotation | Torque | Summary
        Rows: (1) reference vs measurement, (2) errors, (3) outputs
        """
        if not self.control_data:
            print("No data to plot")
            return

        # ---------- gather ----------
        ts = np.array([d["timestamp"] for d in self.control_data])

        # linear: force
        Fm  = np.array([d["force_vector"] for d in self.control_data])                  # measured
        Fr  = np.array([d["reference_force_vector"] for d in self.control_data])        # ref
        Fe  = np.array([d["force_error_vector"] for d in self.control_data])            # error
        Fo  = np.array([d["force_output_vector"] for d in self.control_data])           # output

        # linear: position
        Pm  = np.array([d["position"] for d in self.control_data])
        Pr  = np.array([d["reference_position"] for d in self.control_data])
        Pe  = np.array([d["position_error_vector"] for d in self.control_data])
        Po  = np.array([d["position_output_vector"] for d in self.control_data])

        # orientation: rotation
        has_rotation = "rotation" in self.control_data[0] and "reference_rotation" in self.control_data[0]
        if has_rotation:
            Rm  = np.array([d["rotation"] for d in self.control_data])                  # rotvec (rad)
            Rr  = np.array([d["reference_rotation"] for d in self.control_data])
            Re  = np.array([d["rotation_error_vector"] for d in self.control_data])
            Ro  = np.array([d["rotation_output_vector"] for d in self.control_data])
        else:
            Rm = Rr = Re = Ro = None

        # orientation: torque
        has_torque = "torque_vector" in self.control_data[0]
        if has_torque:
            Tm  = np.array([d["torque_vector"] for d in self.control_data])             # Nm
            Tr  = np.array([d["torque_ref_vector"] for d in self.control_data])         # zeros
            Te  = np.array([d["torque_error_vector"] for d in self.control_data])
            To  = np.array([d["torque_output_vector"] for d in self.control_data])      # rad/s contrib
        else:
            Tm = Tr = Te = To = None

        # totals & summary
        Ltot = np.array([d["total_output_vector"] for d in self.control_data])          # m/s
        Otot = np.array([d["orientation_output_vector"] for d in self.control_data])    # rad/s
        start_p = self.control_data[0].get("start_position", Pm[0])
        dist = np.linalg.norm(Pm - start_p, axis=1)

        # ---------- figure ----------
        # 5 columns: Force | Position | Rotation | Torque | Summary
        n_cols = 5
        fig_w  = 30
        plt.figure(figsize=(fig_w, 12))
        colors = ["red", "green", "blue"]
        axlbl  = ["X", "Y", "Z"]

        def col(row, col_idx):
            """Compute subplot index for (row 1..3, col 1..5)."""
            return (row - 1) * n_cols + col_idx

        # ========== COLUMN 1: FORCE ==========
        # Row 1: ref vs measured
        plt.subplot(3, n_cols, col(1,1))
        for i in range(3):
            plt.plot(ts, Fm[:,i], label=f"{axlbl[i]} Force", linewidth=2, color=colors[i])
            plt.plot(ts, Fr[:,i], "--", label=f"{axlbl[i]} Target", linewidth=2, alpha=0.7, color=colors[i])
        plt.title("Force vs Target"); plt.xlabel("Time (s)"); plt.ylabel("Force (N)")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 2: errors
        plt.subplot(3, n_cols, col(2,1))
        mag = np.linalg.norm(Fe, axis=1)
        for i in range(3): plt.plot(ts, Fe[:,i], label=f"Err {axlbl[i]}", linewidth=2, color=colors[i])
        plt.plot(ts, mag, "--", label="|Err|", linewidth=2, color="black")
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title("Force Error"); plt.xlabel("Time (s)"); plt.ylabel("N")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 3: outputs
        plt.subplot(3, n_cols, col(3,1))
        mag = np.linalg.norm(Fo, axis=1)
        for i in range(3): plt.plot(ts, Fo[:,i], label=f"Out {axlbl[i]}", linewidth=2, color=colors[i])
        plt.plot(ts, mag, "--", label="|Out|", linewidth=2, color="black")
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title("Force PID Output (→ linear)"); plt.xlabel("Time (s)"); plt.ylabel("m/s contrib")
        plt.legend(); plt.grid(True, alpha=0.3)

        # ========== COLUMN 2: POSITION ==========
        # Row 1
        plt.subplot(3, n_cols, col(1,2))
        for i in range(3):
            plt.plot(ts, Pm[:,i], label=f"{axlbl[i]} Pos", linewidth=2, color=colors[i])
            plt.plot(ts, Pr[:,i], "--", label=f"{axlbl[i]} Target", linewidth=2, alpha=0.7, color=colors[i])
        plt.title("Position vs Target"); plt.xlabel("Time (s)"); plt.ylabel("m")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 2
        plt.subplot(3, n_cols, col(2,2))
        mag = np.linalg.norm(Pe, axis=1)
        for i in range(3): plt.plot(ts, Pe[:,i], label=f"Err {axlbl[i]}", linewidth=2, color=colors[i])
        plt.plot(ts, mag, "--", label="|Err|", linewidth=2, color="black")
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        dz = self.control_data[0].get("deadzone_threshold", None)
        if dz is not None:
            plt.axhline(dz,  color="orange", linestyle="dotted", linewidth=2, label=f"±Deadzone")
            plt.axhline(-dz, color="orange", linestyle="dotted", linewidth=2)
        plt.title("Position Error"); plt.xlabel("Time (s)"); plt.ylabel("m")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 3
        plt.subplot(3, n_cols, col(3,2))
        mag = np.linalg.norm(Po, axis=1)
        for i in range(3): plt.plot(ts, Po[:,i], label=f"Out {axlbl[i]}", linewidth=2, color=colors[i])
        plt.plot(ts, mag, "--", label="|Out|", linewidth=2, color="black")
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title("Position PID Output (→ linear)"); plt.xlabel("Time (s)"); plt.ylabel("m/s contrib")
        plt.legend(); plt.grid(True, alpha=0.3)

        # ========== COLUMN 3: ROTATION ==========
        if has_rotation:
            # Row 1
            plt.subplot(3, n_cols, col(1,3))
            for i,lbl in enumerate(["Rx","Ry","Rz"]):
                plt.plot(ts, Rm[:,i], label=f"{lbl}", linewidth=2, color=colors[i])
                plt.plot(ts, Rr[:,i], "--", label=f"{lbl} Target", linewidth=2, alpha=0.7, color=colors[i])
            plt.title("Rotation vs Target (rotvec)"); plt.xlabel("Time (s)"); plt.ylabel("rad")
            plt.legend(); plt.grid(True, alpha=0.3)

            # Row 2
            plt.subplot(3, n_cols, col(2,3))
            mag = np.linalg.norm(Re, axis=1)
            for i,lbl in enumerate(["Rx","Ry","Rz"]):
                plt.plot(ts, Re[:,i], label=f"Err {lbl}", linewidth=2, color=colors[i])
            plt.plot(ts, mag, "--", label="|Err|", linewidth=2, color="black")
            plt.axhline(0, color="black", linewidth=1, alpha=0.5)
            plt.title("Rotation Error"); plt.xlabel("Time (s)"); plt.ylabel("rad")
            plt.legend(); plt.grid(True, alpha=0.3)

            # Row 3
            plt.subplot(3, n_cols, col(3,3))
            mag = np.linalg.norm(Ro, axis=1)
            for i,lbl in enumerate(["Rx","Ry","Rz"]):
                plt.plot(ts, Ro[:,i], label=f"Out {lbl}", linewidth=2, color=colors[i])
            plt.plot(ts, mag, "--", label="|Out|", linewidth=2, color="black")
            plt.axhline(0, color="black", linewidth=1, alpha=0.5)
            plt.title("Rotation PID Output (→ orient)"); plt.xlabel("Time (s)"); plt.ylabel("rad/s contrib")
            plt.legend(); plt.grid(True, alpha=0.3)
        else:
            # placeholders if no rotation logged
            for r in (1,2,3):
                plt.subplot(3, n_cols, col(r,3))
                plt.axis("off")

        # ========== COLUMN 4: TORQUE ==========
        if has_torque:
            # Row 1
            plt.subplot(3, n_cols, col(1,4))
            for i,lbl in enumerate(["Tx","Ty","Tz"]):
                plt.plot(ts, Tm[:,i], label=f"{lbl}", linewidth=2, color=colors[i])
                plt.plot(ts, Tr[:,i], "--", label=f"{lbl} Target", linewidth=2, alpha=0.7, color=colors[i])
            plt.title("Torque vs Target"); plt.xlabel("Time (s)"); plt.ylabel("Nm")
            plt.legend(); plt.grid(True, alpha=0.3)

            # Row 2
            plt.subplot(3, n_cols, col(2,4))
            mag = np.linalg.norm(Te, axis=1)
            for i,lbl in enumerate(["Tx","Ty","Tz"]):
                plt.plot(ts, Te[:,i], label=f"Err {lbl}", linewidth=2, color=colors[i])
            plt.plot(ts, mag, "--", label="|Err|", linewidth=2, color="black")
            plt.axhline(0, color="black", linewidth=1, alpha=0.5)
            plt.title("Torque Error"); plt.xlabel("Time (s)"); plt.ylabel("Nm")
            plt.legend(); plt.grid(True, alpha=0.3)

            # Row 3
            plt.subplot(3, n_cols, col(3,4))
            mag = np.linalg.norm(To, axis=1)
            for i,lbl in enumerate(["Tx","Ty","Tz"]):
                plt.plot(ts, To[:,i], label=f"Out {lbl}", linewidth=2, color=colors[i])
            plt.plot(ts, mag, "--", label="|Out|", linewidth=2, color="black")
            plt.axhline(0, color="black", linewidth=1, alpha=0.5)
            plt.title("Torque PID Output (→ orient)"); plt.xlabel("Time (s)"); plt.ylabel("rad/s contrib")
            plt.legend(); plt.grid(True, alpha=0.3)
        else:
            for r in (1,2,3):
                plt.subplot(3, n_cols, col(r,4))
                plt.axis("off")

        # ========== COLUMN 5: SUMMARY ==========
        # Row 1: total outputs (linear & orientation magnitudes)
        plt.subplot(3, n_cols, col(1,5))
        plt.plot(ts, np.linalg.norm(Ltot, axis=1), label="|Linear Out|", linewidth=2)
        plt.plot(ts, np.linalg.norm(Otot, axis=1), label="|Orient Out|", linewidth=2)
        plt.title("Total Output Magnitudes"); plt.xlabel("Time (s)"); plt.ylabel("m/s & rad/s")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 2: total outputs per-axis (linear only)
        plt.subplot(3, n_cols, col(2,5))
        for i in range(3):
            plt.plot(ts, Ltot[:,i], label=f"Linear {axlbl[i]}", linewidth=2, color=colors[i])
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title("Total Linear Output (per-axis)"); plt.xlabel("Time (s)"); plt.ylabel("m/s")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Row 3: distance moved (and cap)
        plt.subplot(3, n_cols, col(3,5))
        for i in range(3):
            plt.plot(ts, Otot[:, i], label=f"Orient {axlbl[i]}", linewidth=2, color=colors[i])
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.title("Total Orientation Output (per-axis)")
        plt.xlabel("Time (s)")
        plt.ylabel("rad/s")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ---------- title + save ----------
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ctrl_summary = "6DOF + Torque" if (has_rotation and has_torque) else \
                    "6DOF" if has_rotation else \
                    "3DOF" + (" + Torque" if has_torque else "")

        plt.suptitle(
            f"Vector Control Dashboard ({ctrl_summary}) — {current_datetime}",
            fontsize=16, y=0.98
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs("plots", exist_ok=True)
        fname = f"plots/control_dashboard_{current_datetime}_{getattr(self, 'robot_id', 'X')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")
