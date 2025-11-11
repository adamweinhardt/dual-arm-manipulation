import numpy as np
import time
import threading
from control.pid_ff_controller import URForceController
import cvxpy as cp
import pinocchio as pin
from scipy.spatial.transform import Rotation as RR
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import datetime
import os

from utils.utils import _compute_initial_box_frame, _freeze_sparsity


class JointOptimization():
    def __init__(self, robotL, robotR, Hz, trajectory):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz

        self.trajectory_npz = np.load(trajectory)

        self.log_every_s = float(1.0)
        self._ctrl_start_wall = None
        self._last_log_wall = None
        self._win_iters = 0
        self._win_loop_time = 0.0
        self._win_solver_time = 0.0
        self._win_deadline_miss = 0
        self._total_iters = 0
        self._total_solver_time = 0.0
        self._total_deadline_miss = 0

        # limits
        self.joint_pose_limit = np.deg2rad(360)   # ±360°
        self.joint_speed_limit = np.deg2rad(180)   # ±90°/s
        self.joint_accel_limit = np.deg2rad(90)   # ±45°/s²

        self.control_data = []
        self.disable_rotation = True

    # ------------------------ references ------------------------
    def innit(self):
        """
        Load trajectory in the box frame and convert once to WORLD per arm using the shared box frame.
        Naming is consistent:
          p_box, v_box, R_box_ref, w_box
          p_gripper_init_*, R_gripper_init_*
          R_world_box0, R_box_to_gripper_*
        """
        # trajectory (BOX frame)
        self.traj_len = self.trajectory_npz['position'].shape[0]
        p_box      = self.trajectory_npz['position']            # (T,3)
        v_box      = self.trajectory_npz['linear_velocity']     # (T,3)
        R_box_ref  = self.trajectory_npz['rotation_matrices']   # (T,3,3)
        w_box      = self.trajectory_npz['angular_velocity']    # (T,3)

        # initial WORLD TCP poses
        state_L = self.robotL.get_state()
        state_R = self.robotR.get_state()
        p_init_L = np.array(state_L["gripper_world"][:3])
        p_init_R = np.array(state_R["gripper_world"][:3])
        R_init_L = RR.from_rotvec(state_L["pose"][3:6]).as_matrix()
        R_init_R = RR.from_rotvec(state_R["pose"][3:6]).as_matrix()


        # allocate WORLD references
        self.position_ref_L         = np.zeros_like(p_box)
        self.position_ref_R         = np.zeros_like(p_box)
        self.velocity_ref_L         = np.zeros_like(v_box)
        self.velocity_ref_R         = np.zeros_like(v_box)
        self.rotation_ref_L         = np.zeros((self.traj_len,3,3))
        self.rotation_ref_R         = np.zeros((self.traj_len,3,3))
        self.angular_velocity_ref_L = np.zeros_like(w_box)
        self.angular_velocity_ref_R = np.zeros_like(w_box)

        for t in range(self.traj_len):
            # linear: world = R_world_box0 * box + init_offset
            self.position_ref_L[t] = p_init_L + p_box[t]
            self.position_ref_R[t] = p_init_R +  p_box[t]
            self.velocity_ref_L[t] = v_box[t]
            self.velocity_ref_R[t] = v_box[t]

            # orientation: R_world_gripper_ref = R_world_box0 * R_box_ref * R_box_to_gripper
            self.rotation_ref_L[t] = R_box_ref[t] @ R_init_L
            self.rotation_ref_R[t] = R_box_ref[t] @ R_init_R

            # angular velocity to world
            self.angular_velocity_ref_L[t] =  w_box[t]
            self.angular_velocity_ref_R[t] =  w_box[t]

    # ------------------------ QP build ------------------------
    def build_qp(self, dt):
        n_L = n_R = 6
        self.qddot_L_var = cp.Variable(n_L, name="qddot_L")
        self.qddot_R_var = cp.Variable(n_R, name="qddot_R")

        # parameters (update each cycle)
        self.J_L_p     = cp.Parameter((6, n_L), name="J_L")
        self.Jdot_L_p  = cp.Parameter((6, n_L), name="Jdot_L")
        self.qdot_L_p  = cp.Parameter(n_L, name="qdot_L")
        self.a_L_p     = cp.Parameter(6, name="a_L")

        self.J_R_p     = cp.Parameter((6, n_R), name="J_R")
        self.Jdot_R_p  = cp.Parameter((6, n_R), name="Jdot_R")
        self.qdot_R_p  = cp.Parameter(n_R, name="qdot_R")
        self.a_R_p     = cp.Parameter(6, name="a_R")

        self.q_L_p     = cp.Parameter(n_L, name="q_L")
        self.q_R_p     = cp.Parameter(n_R, name="q_R")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(dt * dt))

        # objective: accel tracking for both arms
        xddot_L = self.J_L_p @ self.qddot_L_var + self.Jdot_L_p @ self.qdot_L_p
        xddot_R = self.J_R_p @ self.qddot_R_var + self.Jdot_R_p @ self.qdot_R_p
        e_L = xddot_L - self.a_L_p
        e_R = xddot_R - self.a_R_p
        obj = cp.sum_squares(e_L) + cp.sum_squares(e_R)
        if self.disable_rotation:
            W6 = cp.Constant(np.diag([1, 1, 1, 0, 0, 0]))
            obj = cp.sum_squares(W6 @ e_L) + cp.sum_squares(W6 @ e_R)

        # joint constraints
        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_acc_lim = self.joint_accel_limit

        q_next_L    = self.q_L_p    + self.qdot_L_p * self.dt_c + 0.5 * self.qddot_L_var * self.dt2_c
        q_next_R    = self.q_R_p    + self.qdot_R_p * self.dt_c + 0.5 * self.qddot_R_var * self.dt2_c
        qdot_next_L = self.qdot_L_p + self.qddot_L_var * self.dt_c
        qdot_next_R = self.qdot_R_p + self.qddot_R_var * self.dt_c

        cons = [
            -q_pos_lim <= q_next_L,    q_next_L    <= q_pos_lim,
            -q_pos_lim <= q_next_R,    q_next_R    <= q_pos_lim,
            -q_vel_lim <= qdot_next_L, qdot_next_L <= q_vel_lim,
            -q_vel_lim <= qdot_next_R, qdot_next_R <= q_vel_lim,
            -q_acc_lim <= self.qddot_L_var, self.qddot_L_var <= q_acc_lim,
            -q_acc_lim <= self.qddot_R_var, self.qddot_R_var <= q_acc_lim,
        ]
        if self.disable_rotation:
            cons += [ xddot_L[3:6] == 0 ]
            cons += [ xddot_R[3:6] == 0 ]



        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20000,
            adaptive_rho=True,
            adaptive_rho_interval=45,
            polish=False,
            check_termination=10,
            warm_start=True,
        )

    # ------------------------ control loop ------------------------
    def run(self):
        self.innit()
        dt = 1.0 / self.Hz
        self.control_stop = threading.Event()
        i = 0

        now0 = time.perf_counter()
        self._ctrl_start_wall = now0
        self._last_log_wall = now0

        self.build_qp(dt=dt)

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                if i >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # ---- references (WORLD) ----
                p_ref_L = self.position_ref_L[i]
                p_ref_R = self.position_ref_R[i]
                v_ref_L = self.velocity_ref_L[i]
                v_ref_R = self.velocity_ref_R[i]
                R_ref_L = self.rotation_ref_L[i]
                R_ref_R = self.rotation_ref_R[i]
                w_ref_L = self.angular_velocity_ref_L[i]
                w_ref_R = self.angular_velocity_ref_R[i]

                # ---- LEFT kinematics/dynamics (WORLD) ----
                state_L = self.robotL.get_state()
                p_L = np.array(state_L["gripper_world"][:3])
                v_L = np.array(state_L["speed_world"][:3])
                r_L = np.array(state_L["pose"][3:6])  # rotvec
                R_L = RR.from_rotvec(r_L).as_matrix()
                w_L = np.array(state_L["speed_world"][3:6])

                q_L = self.robotL.get_q()
                qdot_L = self.robotL.get_qdot()

                J_L = self.robotL.get_J(q_L)
                Jdot_L = self.robotL.get_Jdot(q_L, qdot_L)
                M_L = self.robotL.get_M(q_L)
                Lambda_L = self.robotL.get_Lambda(J_L, M_L)
                D_L = self.robotL.get_D(self.robotL.K, Lambda_L)

                # errors
                e_p_L = p_ref_L - p_L
                e_v_L = v_ref_L - v_L
                e_w_L = w_ref_L - w_L
                e_R_L = R_ref_L @ R_L.T
                e_r_L = RR.from_matrix(e_R_L).as_rotvec()

                if self.disable_rotation:
                    e_r_L[:] = 0.0; e_w_L[:] = 0.0
                    w_ref_L = w_L
                    D_L[3:, :] = 0.0
                    D_L[:, 3:] = 0.0

                f_L = self.robotL.get_wrench_desired(D_L, self.robotL.K,
                                                     e_p_L, e_r_L, e_v_L, e_w_L)

                # ---- RIGHT kinematics/dynamics (WORLD) ----
                state_R = self.robotR.get_state()
                p_R = np.array(state_R["gripper_world"][:3])
                v_R = np.array(state_R["speed_world"][:3])
                r_R = np.array(state_R["pose"][3:6])  # rotvec
                R_R = RR.from_rotvec(r_R).as_matrix()
                w_R = np.array(state_R["speed_world"][3:6])

                q_R = self.robotR.get_q()
                qdot_R = self.robotR.get_qdot()

                J_R = self.robotR.get_J(q_R)
                Jdot_R = self.robotR.get_Jdot(q_R, qdot_R)
                M_R = self.robotR.get_M(q_R)
                Lambda_R = self.robotR.get_Lambda(J_R, M_R)
                D_R = self.robotR.get_D(self.robotR.K, Lambda_R)

                # errors
                e_p_R = p_ref_R - p_R
                e_v_R = v_ref_R - v_R
                e_w_R = w_ref_R - w_R
                e_R_R = R_ref_R @ R_R.T
                e_r_R = RR.from_matrix(e_R_R).as_rotvec()

                if self.disable_rotation:
                    e_r_R[:] = 0.0; e_w_R[:] = 0.0
                    w_ref_R = w_R
                    D_R[3:, :] = 0.0
                    D_R[:, 3:] = 0.0

                f_R = self.robotR.get_wrench_desired(D_R, self.robotR.K,
                                                     e_p_R, e_r_R, e_v_R, e_w_R)

                # ---- desired task accelerations (WORLD) for QP ----
                a_L = np.linalg.solve(Lambda_L, f_L)
                a_R = np.linalg.solve(Lambda_R, f_R)
                if self.disable_rotation:
                    a_L[3:] = 0.0
                    a_R[3:] = 0.0

                # set Parameters (freeze sparsity)
                self.J_L_p.value    = _freeze_sparsity(J_L)
                self.Jdot_L_p.value = _freeze_sparsity(Jdot_L)
                self.J_R_p.value    = _freeze_sparsity(J_R)
                self.Jdot_R_p.value = _freeze_sparsity(Jdot_R)

                self.qdot_L_p.value = qdot_L
                self.a_L_p.value    = a_L

                self.qdot_R_p.value = qdot_R
                self.a_R_p.value    = a_R

                self.q_L_p.value    = q_L
                self.q_R_p.value    = q_R

                # solve QP
                _solve_t0 = time.perf_counter()
                try:
                    self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                except Exception:
                    self.qp.status = "error"
                _solve_dt = time.perf_counter() - _solve_t0
                self._win_solver_time += _solve_dt
                self._total_solver_time += _solve_dt

                if self.qp.status not in ("optimal", "optimal_inaccurate"):
                    qddot_L_sol = np.zeros_like(q_L)
                    qddot_R_sol = np.zeros_like(q_R)
                else:
                    qddot_L_sol = np.asarray(self.qddot_L_var.value).reshape(-1)
                    qddot_R_sol = np.asarray(self.qddot_R_var.value).reshape(-1)

                # diagnostics
                if i % 50 == 0:
                    pass

                # integrate solver outputs (for plotting & feed)
                qdot_L_cmd = qdot_L + qddot_L_sol * dt
                qdot_R_cmd = qdot_R + qddot_R_sol * dt
                q_L_cmd    = q_L    + qdot_L * dt + 0.5 * qddot_L_sol * dt * dt
                q_R_cmd    = q_R    + qdot_R * dt + 0.5 * qddot_R_sol * dt * dt

                # send commands
                self.robotL.speedJ(qdot_L_cmd.tolist(), dt)
                self.robotR.speedJ(qdot_R_cmd.tolist(), dt)

                # log
                tcp_log_L = dict(
                    p=p_L, v=v_L, rvec=r_L, w=w_L,
                    p_ref=p_ref_L, v_ref=v_ref_L,
                    rvec_ref=RR.from_matrix(R_ref_L).as_rotvec(),
                    w_ref=w_ref_L,
                    e_p=e_p_L, e_r=e_r_L, e_v=e_v_L, e_w=e_w_L
                )
                tcp_log_R = dict(
                    p=p_R, v=v_R, rvec=r_R, w=w_R,
                    p_ref=p_ref_R, v_ref=v_ref_R,
                    rvec_ref=RR.from_matrix(R_ref_R).as_rotvec(),
                    w_ref=w_ref_R,
                    e_p=e_p_R, e_r=e_r_R, e_v=e_v_R, e_w=e_w_R
                )

                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": self.qp.status,
                    "obj": self.qp.value,
                    # joint space (measured + integrated-from-QP)
                    "q_L": q_L, "qdot_L": qdot_L, "qddot_L": qddot_L_sol,
                    "qdot_cmd_L": qdot_L_cmd, "q_cmd_L": q_L_cmd,
                    "q_R": q_R, "qdot_R": qdot_R, "qddot_R": qddot_R_sol,
                    "qdot_cmd_R": qdot_R_cmd, "q_cmd_R": q_R_cmd,
                    # task space
                    "tcp_L": tcp_log_L,
                    "tcp_R": tcp_log_R,
                    # impedance bits
                    "f_L": f_L, "f_R": f_R,
                })

            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.robotL.speedStop()
                    self.robotR.speedStop()
                except:
                    pass
                break

            # rate-keeping + rolling stats
            elapsed = time.perf_counter() - loop_start
            self._win_loop_time += elapsed
            self._win_iters += 1
            self._total_iters += 1
            if elapsed > dt + 1e-4:
                self._win_deadline_miss += 1
                self._total_deadline_miss += 1

            now = time.perf_counter()
            if now - self._last_log_wall >= self.log_every_s and self._win_iters > 0:
                avg_period = self._win_loop_time / self._win_iters
                avg_hz = (1.0 / avg_period) if avg_period > 0 else float('nan')
                avg_solver_ms = (self._win_solver_time / self._win_iters) * 1000.0
                miss_pct = (100.0 * self._win_deadline_miss / self._win_iters)
                print(f"[CTRL] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms/iter | deadline miss {miss_pct:5.1f}% "
                      f"over {self._win_iters} iters")
                self._win_iters = 0
                self._win_loop_time = 0.0
                self._win_solver_time = 0.0
                self._win_deadline_miss = 0
                self._last_log_wall = now

            time.sleep(max(0, dt - elapsed))
            i += 1

        try:
            self.robotL.speedStop()
            self.robotR.speedStop()
        except:
            pass

        total_time = time.perf_counter() - self._ctrl_start_wall if self._ctrl_start_wall else 0.0
        overall_hz = (self._total_iters / total_time) if total_time > 0 else float('nan')
        overall_solver_ms = (self._total_solver_time / max(self._total_iters, 1)) * 1000.0
        overall_miss_pct = (100.0 * self._total_deadline_miss / max(self._total_iters, 1))
        print(f"[CTRL SUMMARY] {overall_hz:6.2f} Hz over {self._total_iters} iters in {total_time:.2f}s | "
              f"solver avg {overall_solver_ms:.2f} ms | deadline miss {overall_miss_pct:.1f}%")

    # ------------------------ plotting ------------------------
    def plot_taskspace_tracking(self, robots=("L", "R"), title_prefix="TaskspaceTracking"):
        if not self.control_data:
            print("No control_data to plot.")
            return
        os.makedirs("plots", exist_ok=True)

        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        colors = plt.cm.tab10(np.arange(3))  # X,Y,Z
        labels = ["X", "Y", "Z"]

        for side in robots:
            key_tcp = f"tcp_{side}"
            if key_tcp not in self.control_data[0]:
                print(f"Missing {key_tcp}; skipping {side}.")
                continue

            # Raw + ref
            P        = np.array([d[key_tcp]["p"]         for d in self.control_data])       # (T,3)
            V        = np.array([d[key_tcp]["v"]         for d in self.control_data])       # (T,3)
            RVEC     = np.array([d[key_tcp]["rvec"]      for d in self.control_data])       # (T,3)
            W        = np.array([d[key_tcp]["w"]         for d in self.control_data])       # (T,3)
            P_ref    = np.array([d[key_tcp]["p_ref"]     for d in self.control_data])       # (T,3)
            V_ref    = np.array([d[key_tcp]["v_ref"]     for d in self.control_data])       # (T,3)
            RVEC_ref = np.array([d[key_tcp]["rvec_ref"]  for d in self.control_data])       # (T,3)
            W_ref    = np.array([d[key_tcp]["w_ref"]     for d in self.control_data])       # (T,3)

            # Errors (logged)
            E_p = np.array([d[key_tcp]["e_p"] for d in self.control_data])                  # (T,3)
            E_v = np.array([d[key_tcp]["e_v"] for d in self.control_data])                  # (T,3)
            E_w = np.array([d[key_tcp]["e_w"] for d in self.control_data])                  # (T,3)
            E_r = np.array([d[key_tcp]["e_r"] for d in self.control_data])                  # (T,3)

            # Magnitudes (black)
            E_p_mag = np.linalg.norm(E_p, axis=1)
            E_v_mag = np.linalg.norm(E_v, axis=1)
            E_w_mag = np.linalg.norm(E_w, axis=1)
            E_r_mag = np.linalg.norm(E_r, axis=1)

            fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
            fig.suptitle(f"{title_prefix} – Robot {side} – {timestamp_str}", fontsize=14, y=0.99)

            # --- Row 1: Position vs Ref (left) | Translational error (right)
            ax = axes[0, 0]
            for k in range(3):
                ax.plot(t, P[:, k], color=colors[k], label=f"p{labels[k]}")
                ax.plot(t, P_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"p{labels[k]} ref")
            ax.set_title("Position vs Ref"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            ax = axes[0, 1]
            for k in range(3):
                ax.plot(t, E_p[:, k], color=colors[k], label=f"e_p{labels[k]}")
            ax.plot(t, E_p_mag, "-", linewidth=2.0, color="black", label="‖e_p‖")
            ax.set_title("Translational Error"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=4, fontsize=8)

            # --- Row 2: Rotation (rotvec) vs Ref | Rotational error
            ax = axes[1, 0]
            for k in range(3):
                ax.plot(t, RVEC[:, k], color=colors[k], label=f"r{labels[k]}")
                ax.plot(t, RVEC_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"r{labels[k]} ref")
            ax.set_title("Rotation (rotvec) vs Ref"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            ax = axes[1, 1]
            for k in range(3):
                ax.plot(t, E_r[:, k], color=colors[k], label=f"e_r{labels[k]}")
            ax.plot(t, E_r_mag, "-", linewidth=2.0, color="black", label="‖e_r‖")
            ax.set_title("Rotational Error (rotvec of R_ref Rᵀ)"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=4, fontsize=8)

            # --- Row 3: Linear Velocity vs Ref | Linear-velocity error
            ax = axes[2, 0]
            for k in range(3):
                ax.plot(t, V[:, k], color=colors[k], label=f"v{labels[k]}")
                ax.plot(t, V_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"v{labels[k]} ref")
            ax.set_title("Linear Velocity vs Ref"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            ax = axes[2, 1]
            for k in range(3):
                ax.plot(t, E_v[:, k], color=colors[k], label=f"e_v{labels[k]}")
            ax.plot(t, E_v_mag, "-", linewidth=2.0, color="black", label="‖e_v‖")
            ax.set_title("Linear-Velocity Error"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=4, fontsize=8)

            # --- Row 4: Angular Velocity vs Ref | Angular-velocity error
            ax = axes[3, 0]
            for k in range(3):
                ax.plot(t, W[:, k], color=colors[k], label=f"w{labels[k]}")
                ax.plot(t, W_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"w{labels[k]} ref")
            ax.set_title("Angular Velocity vs Ref"); ax.set_xlabel("s"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            ax = axes[3, 1]
            for k in range(3):
                ax.plot(t, E_w[:, k], color=colors[k], label=f"e_w{labels[k]}")
            ax.plot(t, E_w_mag, "-", linewidth=2.0, color="black", label="‖e_w‖")
            ax.set_title("Angular-Velocity Error"); ax.set_xlabel("s"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3)
            ax.legend(ncol=4, fontsize=8)

            plt.tight_layout(rect=[0, 0.035, 1, 0.97])
            fname = f"plots/{title_prefix.lower()}_{side}_{timestamp_str}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved: {fname}")

    def plot_qp_and_jointspace(self, robots=("L", "R"), title_prefix="QP_and_Jointspace"):
        """
        Plots:
          - q (measured) vs q_cmd (integrated-from-QP)
          - qdot (measured) vs qdot_cmd
          - qddot (QP)
          - feasibility flag
        """
        if not self.control_data:
            print("No control_data to plot.")
            return
        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        obj = np.array([d.get("obj", np.nan) for d in self.control_data])
        status = [d.get("status", "") for d in self.control_data]

        plt.figure(figsize=(12, 4))
        plt.plot(t, obj, label="QP objective")
        bad = np.array([ii for ii, s in enumerate(status) if s not in ("optimal", "optimal_inaccurate")], dtype=int)
        if bad.size:
            plt.scatter(t[bad], obj[bad], marker="x", s=30, label="non-optimal", zorder=3)
        plt.title(f"QP Objective over Time – {timestamp_str}")
        plt.xlabel("time [s]"); plt.ylabel("objective"); plt.grid(True, alpha=0.3); plt.legend()
        fname_obj = f"plots/{title_prefix.lower()}_objective_{timestamp_str}.png"
        plt.tight_layout(); plt.savefig(fname_obj, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname_obj}")

        for side in robots:
            sample = self.control_data[0]
            q_key, qdot_key, qddot_key = f"q_{side}", f"qdot_{side}", f"qddot_{side}"
            qdot_cmdkey, q_cmdkey = f"qdot_cmd_{side}", f"q_cmd_{side}"
            missing = [k for k in (q_key, qdot_key, qddot_key, qdot_cmdkey, q_cmdkey) if k not in sample]
            if missing:
                print(f"Missing {missing} for {side}; skipping.")
                continue

            Q         = np.vstack([d[q_key]        for d in self.control_data])
            QDOT      = np.vstack([d[qdot_key]     for d in self.control_data])
            QDDOT     = np.vstack([d[qddot_key]    for d in self.control_data])
            QDOT_cmd  = np.vstack([d[qdot_cmdkey]  for d in self.control_data])
            Q_cmd     = np.vstack([d[q_cmdkey]     for d in self.control_data])

            n = Q.shape[1]
            jlabels = [f"J{i+1}" for i in range(n)]

            fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
            fig.suptitle(f"{title_prefix} – Robot {side} – {timestamp_str}", fontsize=14, y=0.98)

            # q measured vs q_cmd (integrated from QP)
            ax = axes[0, 0]
            for j in range(n):
                ax.plot(t, Q[:, j], label=f"{jlabels[j]} meas")
                ax.plot(t, Q_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd(int)")
            ax.set_title("Joint Positions: measured vs integrated-from-QP")
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(2*n, 6), fontsize=8)

            # qdot measured vs qdot_cmd
            ax = axes[0, 1]
            for j in range(n):
                ax.plot(t, QDOT[:, j], label=f"{jlabels[j]} meas")
                ax.plot(t, QDOT_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd")
            ax.set_title("Joint Velocities: measured vs commanded")
            ax.set_ylabel("rad/s")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(2*n, 6), fontsize=8)

            # qddot (QP)
            ax = axes[1, 0]
            for j in range(n):
                ax.plot(t, QDDOT[:, j], label=jlabels[j])
            ax.set_title("Joint Accelerations qddot (QP)")
            ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(n, 6), fontsize=8)

            # feasibility flag
            ax = axes[1, 1]
            y = np.array([0 if s in ("optimal", "optimal_inaccurate") else 1 for s in status], dtype=float)
            ax.step(t, y, where="post")
            ax.set_title("QP infeasible/non-optimal")
            ax.set_xlabel("time [s]")
            ax.set_yticks([0, 1]); ax.set_yticklabels(["ok", "bad"])
            ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"plots/{title_prefix.lower()}_{side}_{timestamp_str}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved: {fname}")


class URImpedanceController(URForceController):
    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = K
        self.D = np.zeros((6, 6))

        self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("tool0")

        self.q = self.get_q()
        self.qdot = self.get_qdot()
        self.J = self.get_J(self.q)
        self.Jdot = self.get_Jdot(self.q, self.qdot)
        self.M = self.get_M(self.q)
        self.Lambda = self.get_Lambda(self.J, self.M)
        self.D = self.get_D(self.K, self.Lambda)

        self.cotrol_data = []

    def get_q(self):
        return np.array(self.rtde_receive.getActualQ(), dtype=float)

    def get_qdot(self):
        return np.array(self.rtde_receive.getActualQd(), dtype=float)

    def get_J(self, q):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.pin_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(J)

    def get_Jdot(self, q, v):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model, self.pin_data, self.pin_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(dJ)

    def get_M(self, q):
        M = pin.crba(self.pin_model, self.pin_data, q)
        return np.asarray(0.5 * (M + M.T))

    def get_Lambda(self, J, M, lam2=1e-3):
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        A = 0.5 * (A + A.T)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_dinv = S / (S**2 + lam2)
        Lam = (Vt.T * S_dinv) @ U.T
        return 0.5 * (Lam + Lam.T)

    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda); S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        return np.real_if_close(0.5 * (D + D.T))

    def get_wrench_desired(self, D, K, e_p, e_r, e_v, e_w):
        Xd_err = np.hstack((e_v, e_w))
        X_err = np.hstack((e_p, e_r))
        wrench = D @ Xd_err + K @ X_err
        return wrench

