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
from control.PID.pid_controller import flip_y_component_in_rotations, flip_x_component_in_rotations

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
    
    def update_weights(self, kp=None, ki=None, kd=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

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
        Kp_p=0,
        Ki_p=0,
        Kd_p=0,
    ):
        super().__init__(ip)

        self.control_active = False
        self.lifting = False
        self.control_thread = None
        self.control_stop = threading.Event()

        self.kp_f=kp_f
        self.ki_f=ki_f
        self.kd_f=kd_f

        self.control_rate_hz = hz
        self.force_pid = VectorPIDController(kp=kp_f, ki=ki_f, kd=kd_f, dt=1 / hz)
        self.pose_pid = VectorPIDController(kp=kp_p, ki=ki_p, kd=kd_p, dt=1 / hz)
        self.rot_pid = VectorPIDController(kp=kp_r, ki=ki_r, kd=kd_r, dt=1 / hz)
        self.ff_pose_pid = VectorPIDController(kp=Kp_p, ki=Ki_p, kd=Kd_p, dt=1 / hz)
        self.control_data = []

        # ZMQ
        self.grasping_context = zmq.Context()
        self.grasping_socket = self.grasping_context.socket(zmq.SUB)
        self.grasping_socket.setsockopt(zmq.CONFLATE, 1)
        self.grasping_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        grasping_port = 5560
        self.grasping_socket.connect(f"tcp://127.0.0.1:{grasping_port}")

        self.current_grasping_data = {}
        self.box_data = []

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
    
    def get_box_data(self):
        while not self._update_grasping_data():
            time.sleep(0.01)

        box_id = list(self.current_grasping_data.keys())[0]
        g = self.current_grasping_data[box_id]

        box_position = g.get("box_position")
        box_rotation_matrix = g.get("box_rotation_matrix")


        if box_position is None or box_rotation_matrix is None:
            return None, None

        return (
            np.array(box_position),
            np.array(box_rotation_matrix)
        )
    
    def get_grasping_data_both(self):
        while not self._update_grasping_data():
            time.sleep(0.01)

        box_id = list(self.current_grasping_data.keys())[0]
        g = self.current_grasping_data[box_id]

        grasping_point0 = g.get("grasping_point0")
        approach_point0 = g.get("approach_point0")
        normal_vector0 = g.get("normal0")
        grasping_point1 = g.get("grasping_point1")
        approach_point1 = g.get("approach_point1")
        normal_vector1 = g.get("normal1")


        return (
            np.array(grasping_point0),
            np.array(grasping_point1),
            np.array(approach_point0),
            np.array(approach_point1),
            np.array(normal_vector0),
            np.array(normal_vector1),
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
        R_ref = R_WB @ self._R_BG

        return R_ref
    

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
        self.reference_linear_velocity = np.array([0,0,0], dtype=float)
        self.reference_angular_velocity = np.array([0,0,0], dtype=float)

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
            if self.robot_id == 0:
                print("Flipped for robot 0")
                self.rot_updates = flip_y_component_in_rotations(self.rot_updates)
                self.rot_updates = flip_x_component_in_rotations(self.rot_updates)
            self.reference_linear_velocity_updates = traj_npz.get("linear_velocity", None)
            self.reference_angular_velocity_updates = traj_npz.get("angular_velocity", None)
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
            self.reference_linear_velocity = None
            self.reference_angular_velocity = None
            self._traj_len = 0


        # Reset PIDs & logs
        self.force_pid.reset()
        self.pose_pid.reset()
        self.rot_pid.reset()
        self.ff_pose_pid.reset()
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
                if trajectory_index >= self._traj_len:
                    print("Trajectory completed.")
                    break
                self.box_data.append(self.get_box_data())
                if not trajectory_started:
                        reference_force = 50 # 150
                        base_force = 12.5
                        factor = base_force / reference_force

                        kp_f = 0.003 * factor
                        ki_f = 0.0000 * factor
                        kd_f = 0.0015 * factor
                        self.force_pid.update_weights(kp=kp_f, ki=ki_f, kd=kd_f)

                state = self.get_state()
                current_force_vector = np.array(state["filtered_force_world"][:3])
                current_position = np.array(state["gripper_world"][:3])
                current_rotation = np.array(state["pose"][3:6])  # rotvec
                current_rotation_matrix = R.from_rotvec(current_rotation).as_matrix()
                current_linear_velocity = np.array(state["speed_world"])[:3]
                current_angular_velocity = np.array(state["speed_world"])[3:6]

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
                    self.force_pid.update_weights(kp=self.kp_f, ki=self.ki_f, kd=self.kd_f)


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

                    self.reference_linear_velocity = self.reference_linear_velocity_updates[trajectory_index]
                    self.reference_angular_velocity = self.reference_angular_velocity_updates[trajectory_index]

                    if self.robot_id == 0:
                        offset = [-0.055, 0, 0]
                    else:
                        offset = [0.055, 0, 0]

                    r_B_plus_offset = self._r_B + offset

                    delta_p_rot_B = (R_B0B - np.eye(3)) @ r_B_plus_offset
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

                    if getattr(self, "traj_linear_velocity", None) is not None:
                        if trajectory_index < len(self.traj_linear_velocity):
                            self.reference_linear_velocity = np.array(
                                self.traj_linear_velocity[trajectory_index], dtype=float
                            )

                    if getattr(self, "traj_angular_velocity", None) is not None:
                        if trajectory_index < len(self.traj_angular_velocity):
                            self.reference_angular_velocity = np.array(
                                self.traj_angular_velocity[trajectory_index], dtype=float
                            )


                    trajectory_index += 1

                # =========================== PID Control ===========================
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
                # feedforward position
                ff_position_output_vector, ff_pos_p_term, ff_pos_i_term, ff_pos_d_term = (
                    self.ff_pose_pid.update(position_error_vector)
                )
                position_output_vector, pos_p_term, pos_i_term, pos_d_term = (
                    self.pose_pid.update(position_error_vector)
                )
                # DEADZONE
                if self.deadzone_threshold is not None or self.deadzone_threshold !=0:
                    r = float(self.deadzone_threshold)
                    shaped_err = np.zeros(3)

                    # --- Smooth deadzone using offset method ---
                    for i in range(3):
                        e = position_error_vector[i]
                        if abs(e) <= r:
                            shaped_err[i] = 0.0
                        elif e > 0:
                            shaped_err[i] = e - r
                        else:
                            shaped_err[i] = e + r
                    position_output_vector = shaped_err
                else:
                    position_output_vector = position_output_vector

                total_output_vector = -force_output_vector + position_output_vector + ff_position_output_vector

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

                # Command
                speed_command = [*total_output_vector, *rotation_output_vector]
                self.speedL_world(speed_command, acceleration=0.18, time_duration=0.001)

                # --- Log ---
                self.control_data.append(
                    {
                        "timestamp": time.time() - self.start_time,
                        "force_vector": current_force_vector,
                        "position": current_position,
                        "rotation": current_rotation,
                        "linear_velocity": current_linear_velocity,
                        "angular_velocity": current_angular_velocity,
                        "reference_position": self.reference_position.copy(),
                        "reference_rotation": self.reference_rotation.copy(),
                        "reference_linear_velocity": self.reference_linear_velocity.copy(),      # from trajectory
                        "reference_angular_velocity": self.reference_angular_velocity.copy(),
                        "reference_force_vector": ref_force_vector,
                        "position_error_vector": position_error_vector,
                        "rotation_error_vector": rotation_error_vector,
                        "force_error_vector": force_error_vector,

                        # outputs
                        "position_output_vector": position_output_vector,
                        "rotation_output_vector": rotation_output_vector,
                        "force_output_vector": force_output_vector,
                        "ff_position_output_vector": ff_position_output_vector,   
                        "total_output_vector": total_output_vector,

                        # PID terms
                        "force_p_term": force_p_term,
                        "force_i_term": force_i_term,
                        "force_d_term": force_d_term,
                        "pos_p_term": pos_p_term,
                        "pos_i_term": pos_i_term,
                        "pos_d_term": pos_d_term,
                        "rot_p_term": rot_p_term,
                        "rot_i_term": rot_i_term,
                        "rot_d_term": rot_d_term,

                        # Feed-forward Pose PID terms
                        "ff_pos_p_term": ff_pos_p_term,      
                        "ff_pos_i_term": ff_pos_i_term,      
                        "ff_pos_d_term": ff_pos_d_term,       

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
        if not self.control_data:
            print("No data to plot")
            return

        # Check required keys exist in the log
        required_force = {"force_p_term", "force_i_term", "force_d_term"}
        required_pose  = {"pos_p_term", "pos_i_term", "pos_d_term"}
        required_ff    = {"ff_pos_p_term", "ff_pos_i_term", "ff_pos_d_term"}

        have_force = all(k in self.control_data[0] for k in required_force)
        have_pose  = all(k in self.control_data[0] for k in required_pose)
        have_ff    = all(k in self.control_data[0] for k in required_ff)

        if not (have_force and have_pose and have_ff):
            missing = []
            if not have_force: missing.append("Force PID terms")
            if not have_pose:  missing.append("Pose PID terms")
            if not have_ff:    missing.append("Feed-Forward Pose PID terms")
            print("PID components missing in control_data:", ", ".join(missing))
            return

        import numpy as np
        import os, datetime
        import matplotlib.pyplot as plt

        # ---------- Extract ----------
        T = np.array([d["timestamp"] for d in self.control_data])

        force_p = np.array([d["force_p_term"] for d in self.control_data])  # (N,3)
        force_i = np.array([d["force_i_term"] for d in self.control_data])
        force_d = np.array([d["force_d_term"] for d in self.control_data])

        pose_p  = np.array([d["pos_p_term"] for d in self.control_data])
        pose_i  = np.array([d["pos_i_term"] for d in self.control_data])
        pose_d  = np.array([d["pos_d_term"] for d in self.control_data])

        ff_p    = np.array([d["ff_pos_p_term"] for d in self.control_data])
        ff_i    = np.array([d["ff_pos_i_term"] for d in self.control_data])
        ff_d    = np.array([d["ff_pos_d_term"] for d in self.control_data])

        # ---------- Summed terms (Force + Pose + Feed-Forward) ----------
        sum_p = force_p + pose_p + ff_p
        sum_i = force_i + pose_i + ff_i
        sum_d = force_d + pose_d + ff_d

        # ---------- Pretty strings for gains ----------
        def gains_str(pid_obj, label):
            if pid_obj is None:
                return f"{label}: (n/a)"
            kp, ki, kd = pid_obj.kp, pid_obj.ki, pid_obj.kd
            if np.isscalar(kp):
                return f"{label}: (Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f})"
            return f"{label}: (Kp={kp}, Ki={ki}, Kd={kd})"

        force_gains = gains_str(getattr(self, "force_pid", None), "Force PID")
        pose_gains  = gains_str(getattr(self, "pose_pid", None),  "Pose PID")
        ff_gains    = gains_str(getattr(self, "ff_pose_pid", None),"FF Pose PID")

        # ---------- Plot ----------
        colors = ["red", "green", "blue"]
        axis_labels = ["X", "Y", "Z"]

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig = plt.figure(figsize=(28, 14))
        plt.suptitle(
            "PID Components (Force / Pose / Feed-Forward Pose) and Summed Terms\n"
            f"{force_gains} | {pose_gains} | {ff_gains}\n"
            f"Generated: {current_datetime}",
            fontsize=16, y=0.98
        )

        def plot_block(row, col_offset, data, title):
            """Plot a 3-axis time series block at (row, col), using shared colors."""
            idx = (row - 1) * 4 + col_offset
            plt.subplot(3, 4, idx)
            for a in range(3):
                plt.plot(T, data[:, a], label=f"{axis_labels[a]}", linewidth=2, color=colors[a])
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel("Output")
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            plt.legend()

        # Row 1: P terms
        plot_block(1, 1, force_p, "Force PID — P")
        plot_block(1, 2, pose_p,  "Pose PID — P")
        plot_block(1, 3, ff_p,    "FF Pose PID — P")
        plot_block(1, 4, sum_p,   "SUM P = Force + Pose + FF")

        # Row 2: I terms
        plot_block(2, 1, force_i, "Force PID — I")
        plot_block(2, 2, pose_i,  "Pose PID — I")
        plot_block(2, 3, ff_i,    "FF Pose PID — I")
        plot_block(2, 4, sum_i,   "SUM I = Force + Pose + FF")

        # Row 3: D terms
        plot_block(3, 1, force_d, "Force PID — D")
        plot_block(3, 2, pose_d,  "Pose PID — D")
        plot_block(3, 3, ff_d,    "FF Pose PID — D")
        plot_block(3, 4, sum_d,   "SUM D = Force + Pose + FF")

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        # Save
        os.makedirs("plots", exist_ok=True)
        robot_id = getattr(self, "robot_id", "x")
        filename = f"plots/pid_force_pose_ff_terms_{current_datetime}_{robot_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PID plot saved: {filename}")

    def plot_data3D(self):
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
        ff_position_output_vectors = np.array(  # <<< NEW
            [d["ff_position_output_vector"] for d in self.control_data]
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
        ff_position_output_magnitudes = np.linalg.norm(ff_position_output_vectors, axis=1)  # <<< NEW
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

        # Position Output Vectors (3D) + Feedforward
        plt.subplot(3, n_cols, 2 * n_cols + 2)
        # PID outputs
        plt.plot(
            timestamps,
            position_output_vectors[:, 0],
            label="PID Out X",
            linewidth=2,
            color="red",
        )
        plt.plot(
            timestamps,
            position_output_vectors[:, 1],
            label="PID Out Y",
            linewidth=2,
            color="green",
        )
        plt.plot(
            timestamps,
            position_output_vectors[:, 2],
            label="PID Out Z",
            linewidth=2,
            color="blue",
        )

        # Feedforward outputs (different bright colors)
        plt.plot(
            timestamps,
            ff_position_output_vectors[:, 0],
            label="FF Out X",
            linewidth=2,
            color="magenta",
        )
        plt.plot(
            timestamps,
            ff_position_output_vectors[:, 1],
            label="FF Out Y",
            linewidth=2,
            color="orange",
        )
        plt.plot(
            timestamps,
            ff_position_output_vectors[:, 2],
            label="FF Out Z",
            linewidth=2,
            color="cyan",
        )

        # Magnitudes
        plt.plot(
            timestamps,
            position_output_magnitudes,
            label="|PID Out|",
            linewidth=2,
            color="black",
        )
        plt.plot(
            timestamps,
            ff_position_output_magnitudes,
            label="|FF Out|",
            linewidth=2,
            color="purple",
        )

        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.title("3D Position Control Outputs (PID + Feedforward)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position Output")
        plt.legend(ncol=2, fontsize=9)
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

    def save_everything(self, out_path=None):
        """
        ONE-SHOT DUMP OF THE ENTIRE RUN TO A SINGLE .NPZ

        Contents include:
        - Time series from self.control_data (timestamps, forces, poses, refs, errors, PID outputs/terms, totals...)
        - Box pose-estimation measurements from self.box_data
        - Reference signals (position, rotation, velocity, angular velocity, force)
        - Frame info (R_WB0, p_WB0, R_BG, r_B) when available
        - Controller configuration (PID gains, rate, caps, thresholds)
        - Start pose, grasp points, trajectory metadata
        - Any plot images saved during the run
        """
        import numpy as np, os, datetime

        # ---------- filename ----------
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        if out_path is None:
            robot_id = getattr(self, "robot_id", "x")
            out_path = os.path.join("logs", f"run_{ts}_r{robot_id}.npz")

        # ---------- control_data stacker ----------
        cd = self.control_data if hasattr(self, "control_data") and self.control_data else []
        stacked = {}
        if cd:
            common_keys = set(cd[0].keys())
            for d in cd[1:]:
                common_keys &= set(d.keys())

            priority = [
                "timestamp",
                # measured
                "force_vector", "position", "rotation",
                "linear_velocity", "angular_velocity",
                # references
                "reference_force_vector", "reference_position", "reference_rotation",
                "reference_linear_velocity", "reference_angular_velocity",
                # errors
                "force_error_vector", "position_error_vector", "rotation_error_vector",
                # outputs
                "force_output_vector", "position_output_vector", "ff_position_output_vector",
                "total_output_vector", "rotation_output_vector",
                # PID terms
                "force_p_term", "force_i_term", "force_d_term",
                "pos_p_term", "pos_i_term", "pos_d_term",
                "ff_pos_p_term", "ff_pos_i_term", "ff_pos_d_term",
                "rot_p_term", "rot_i_term", "rot_d_term",
            ]

            ordered_keys = [k for k in priority if k in common_keys] + \
                        [k for k in sorted(common_keys) if k not in priority]

            for k in ordered_keys:
                try:
                    arr = np.array([d[k] for d in cd])
                    if k != "timestamp" and arr.ndim == 1:
                        arr = np.expand_dims(arr, -1)
                    stacked[k] = arr
                except Exception:
                    pass

        # ---------- box pose-estimation data ----------
        box_positions, box_rotmats = [], []
        if hasattr(self, "box_data") and self.box_data:
            for tpl in self.box_data:
                if tpl is None or len(tpl) < 2 or tpl[0] is None or tpl[1] is None:
                    box_positions.append(np.array([np.nan, np.nan, np.nan]))
                    box_rotmats.append(np.full((3, 3), np.nan))
                else:
                    bp, br = tpl
                    box_positions.append(np.array(bp).reshape(3,))
                    br_arr = np.array(br)
                    if br_arr.size == 9:
                        br_arr = br_arr.reshape(3, 3)
                    else:
                        br_arr = np.full((3, 3), np.nan)
                    box_rotmats.append(br_arr)

        box_positions = np.vstack(box_positions) if box_positions else np.zeros((0, 3))
        box_rotmats   = np.stack(box_rotmats)    if box_rotmats   else np.zeros((0, 3, 3))

        # ---------- frame and reference info ----------
        R_WB0 = getattr(self, "_R_WB0", None)
        p_WB0 = getattr(self, "_p_WB0", None)
        R_BG  = getattr(self, "_R_BG",  None)
        r_B   = getattr(self, "_r_B",   None)

        ref_pos    = getattr(self, "reference_position", None)
        ref_rot    = getattr(self, "reference_rotation", None)
        ref_Rmat   = getattr(self, "reference_rotation_matrix", None)
        ref_force  = getattr(self, "ref_force", None)
        ctrl_dir   = getattr(self, "control_direction", None)

        ref_vel    = getattr(self, "reference_linear_velocity", None)
        ref_omega  = getattr(self, "reference_angular_velocity", None)

        # ---------- controller metadata ----------
        meta = {}

        for label, obj in [
            ("force_pid", getattr(self, "force_pid", None)),
            ("pose_pid", getattr(self, "pose_pid", None)),
            ("rot_pid", getattr(self, "rot_pid", None)),
            ("ff_pose_pid", getattr(self, "ff_pose_pid", None))
        ]:
            if obj is not None:
                try:
                    meta[f"{label}_kp"] = np.array(obj.kp)
                    meta[f"{label}_ki"] = np.array(obj.ki)
                    meta[f"{label}_kd"] = np.array(obj.kd)
                except Exception:
                    pass

        for n in ["kp_f", "ki_f", "kd_f"]:
            if hasattr(self, n):
                meta[n] = float(getattr(self, n))

        Hz_val = float(getattr(self, "control_rate_hz", 0.0))
        meta["control_rate_hz"] = Hz_val
        meta["Hz"] = Hz_val

        meta["distance_cap"] = float(getattr(self, "distance_cap", np.nan)) \
                            if hasattr(self, "distance_cap") else np.nan
        meta["timeout_s"] = float(getattr(self, "control_timeout", np.nan)) \
                            if hasattr(self, "control_timeout") else np.nan
        meta["deadzone_threshold"] = float(getattr(self, "deadzone_threshold", np.nan)) \
                                    if hasattr(self, "deadzone_threshold") else np.nan

        meta["start_time_epoch"] = float(getattr(self, "start_time", np.nan))
        meta["robot_id"] = getattr(self, "robot_id", -1)

        for name in ["start_position", "start_rotation", "grasping_point", "other_robot_grasp_point"]:
            if hasattr(self, name) and getattr(self, name) is not None:
                meta[name] = np.array(getattr(self, name), dtype=float)

        # ---------- include plot paths ----------
        plot_paths = []
        try:
            if os.path.isdir("plots"):
                for fn in os.listdir("plots"):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                        plot_paths.append(os.path.join("plots", fn))
        except Exception:
            pass
        plot_paths_arr = np.array(plot_paths, dtype=object)

        # ---------- pack everything ----------
        pack = {}
        pack.update(stacked)
        pack["box_positions"] = box_positions
        pack["box_rotmats"]   = box_rotmats

        if ref_pos is not None:  pack["ref_position_last"] = np.array(ref_pos)
        if ref_rot is not None:  pack["ref_rotation_last"] = np.array(ref_rot)
        if ref_Rmat is not None: pack["ref_rotation_matrix_last"] = np.array(ref_Rmat)

        # last ref velocities snapshot
        if ref_vel is not None:   pack["ref_linear_velocity_last"] = np.array(ref_vel)
        if ref_omega is not None: pack["ref_angular_velocity_last"] = np.array(ref_omega)

        if ref_force is not None:
            try:
                pack["ref_force"] = np.array(ref_force)
            except Exception:
                pack["ref_force"] = np.array([float(ref_force)])

        if ctrl_dir is not None:
            pack["control_direction"] = np.array(ctrl_dir)

        # frames
        if R_WB0 is not None: pack["frame_R_WB0"] = np.array(R_WB0)
        if p_WB0 is not None: pack["frame_p_WB0"] = np.array(p_WB0)
        if R_BG  is not None: pack["frame_R_BG"]  = np.array(R_BG)
        if r_B   is not None: pack["frame_r_B"]   = np.array(r_B)

        # meta fields
        for k, v in meta.items():
            pack[f"meta__{k}"] = v if isinstance(v, (float, int, np.floating)) else np.array(v)

        pack["plot_paths"] = plot_paths_arr

        # ---------- optional trajectory references ----------
        if hasattr(self, "rot_updates") and self.rot_updates is not None:
            try:
                pack["traj_rotation_matrices"] = np.array(self.rot_updates)
            except Exception:
                pass

        if hasattr(self, "pose_updates_stream") and self.pose_updates_stream is not None:
            try:
                pack["traj_position_offsets"] = np.array(self.pose_updates_stream)
            except Exception:
                pass

        # ---------- derived box reference poses ----------
        try:
            if (hasattr(self, "_R_WB0") and self._R_WB0 is not None and
                hasattr(self, "_p_WB0") and self._p_WB0 is not None and
                hasattr(self, "rot_updates") and self.rot_updates is not None):

                R_WB0 = np.array(self._R_WB0).reshape(3, 3)
                p_WB0 = np.array(self._p_WB0).reshape(3,)

                rot_updates = np.array(self.rot_updates)
                pos_off = None
                if getattr(self, "pose_updates_stream", None) is not None:
                    pos_off = np.array(self.pose_updates_stream)

                K = len(rot_updates)
                box_ref_rotmats   = np.zeros((K, 3, 3))
                box_ref_positions = np.zeros((K, 3))

                for i in range(K):
                    R_B0B = rot_updates[i]
                    R_WB = R_WB0 @ R_B0B
                    box_ref_rotmats[i] = R_WB

                    dp = pos_off[i] if (pos_off is not None and i < len(pos_off)) else np.zeros(3)
                    box_ref_positions[i] = p_WB0 + (R_WB0 @ dp)

                pack["box_ref_rotmats"]   = box_ref_rotmats
                pack["box_ref_positions"] = box_ref_positions
        except Exception:
            pass

        np.savez_compressed(out_path, **pack)
        print(f"[SAVED] {out_path}")
        return out_path

