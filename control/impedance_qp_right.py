import numpy as np
import time
import threading
import cvxpy as cp
import pinocchio as pin
from scipy.spatial.transform import Rotation as RR
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import datetime
import os

from control.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity


class JointOptimizationSingleArm():
    def __init__(self, robot, Hz, trajectory):
        self.robot = robot
        self.Hz = Hz

        self.trajectory_npz = np.load(trajectory)

        # perf stats
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

        # joint limits (rad, rad/s, rad/s^2)
        self.joint_pose_limit  = np.deg2rad(360)   # ±360°
        self.joint_speed_limit = np.deg2rad(180)   # ±180°/s
        self.joint_accel_limit = np.deg2rad(500)   # ±500°/s²

        # data log buffer
        self.control_data = []

    # ------------------------ references ------------------------
    def innit(self):
        """
        Use trajectory deltas in WORLD frame.
        Keep the initial tool orientation fixed for now.
        """
        self.traj_len = self.trajectory_npz['position'].shape[0]

        p_box  = self.trajectory_npz['position']            # (T,3), WORLD deltas
        v_box  = self.trajectory_npz['linear_velocity']     # (T,3), WORLD linear v
        R_box  = self.trajectory_npz['rotation_matrices']   # (T,3,3) (not used directly yet)
        w_box  = self.trajectory_npz['angular_velocity']    # (T,3)

        # initial world pose/orientation
        state = self.robot.get_state()
        p_gripper_init = np.array(state["gripper_world"][:3])
        R_gripper_init = RR.from_rotvec(state["pose"][3:6]).as_matrix()

        # allocate reference arrays
        self.position_ref         = np.zeros_like(p_box)
        self.velocity_ref         = np.zeros_like(v_box)
        self.rotation_ref         = np.zeros((self.traj_len, 3, 3))
        self.angular_velocity_ref = np.zeros_like(w_box)

        for t in range(self.traj_len):
            self.position_ref[t]         = p_gripper_init + p_box[t]
            self.velocity_ref[t]         = v_box[t]
            self.rotation_ref[t]         = R_gripper_init  # lock orientation
            self.angular_velocity_ref[t] = w_box[t]

    # ------------------------ QP build ------------------------
    def build_qp(self, dt):
        """
        QP we solve each tick:
            min || W_task * (xddot_actual - xddot_des) ||^2  +  lambda_reg * ||qddot||^2
            s.t. joint pos/vel/acc bounds (1-step lookahead)

        Where:
        1. We FIRST rescale translation vs rotation accel tracking error so they're numerically comparable.
           (scale_lin, scale_rot)
        2. We THEN apply task preference weights for how much we actually care.
           (w_lin, w_rot)

        Final per-axis weight = scale_* * w_*
        """

        n = 6
        self.qddot_var = cp.Variable(n, name="qddot")

        # parameters that we update every control iteration
        self.J_p     = cp.Parameter((6, n), name="J")
        self.Jdot_p  = cp.Parameter((6, n), name="Jdot")
        self.qdot_p  = cp.Parameter(n,      name="qdot")
        self.a_p     = cp.Parameter(6,      name="a")   # desired task accel
        self.q_p     = cp.Parameter(n,      name="q")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(dt * dt))

        # task-space accel tracking error:
        # xddot_actual = J qddot + Jdot qdot
        xddot_actual = self.J_p @ self.qddot_var + self.Jdot_p @ self.qdot_p
        e_acc = xddot_actual - self.a_p   # [ax, ay, az, alpha_x, alpha_y, alpha_z]

        # ----------------------------
        scale_lin = 4   # ~1/(0.05 m/s^2)
        scale_rot = 4    # ~1/(0.2 rad/s^2)

        w_lin = 1e5
        w_rot = 4e-6
        self.lambda_reg = 1e-6

        # stash for plotting/debug/metadata
        self.scale_lin = scale_lin
        self.scale_rot = scale_rot
        self.w_lin     = w_lin
        self.w_rot     = w_rot

        # final per-axis diagonal weights for the task error (length 6)
        # translation axes [0:3] share same weight; rotation axes [3:6] share same weight
        final_lin_weight = scale_lin * w_lin      # just a number
        final_rot_weight = scale_rot * w_rot      # just a number

        self.final_lin_weight = final_lin_weight
        self.final_rot_weight = final_rot_weight

        W_task_vec_np = np.array([
            final_lin_weight,
            final_lin_weight,
            final_lin_weight,
            final_rot_weight,
            final_rot_weight,
            final_rot_weight,
        ], dtype=float)

        # cvxpy Constant so we can multiply elementwise with e_acc
        self.W_task_vec_c = cp.Constant(W_task_vec_np)

        # regularization on joint accel to avoid crazy jerks

        obj_task = cp.sum_squares(cp.multiply(self.W_task_vec_c, e_acc))
        obj_reg  = self.lambda_reg * cp.sum_squares(self.qddot_var)
        obj      = obj_task + obj_reg

        # ---- joint constraints ----
        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_acc_lim = self.joint_accel_limit

        q_next    = self.q_p    + self.qdot_p * self.dt_c + 0.5 * self.qddot_var * self.dt2_c
        qdot_next = self.qdot_p + self.qddot_var * self.dt_c

        cons = [
            -q_pos_lim <= q_next,    q_next    <= q_pos_lim,
            -q_vel_lim <= qdot_next, qdot_next <= q_vel_lim,
            -q_acc_lim <= self.qddot_var, self.qddot_var <= q_acc_lim,
        ]

        self.qp = cp.Problem(cp.Minimize(obj), cons)

        # solver settings (OSQP)
        self.qp_kwargs = dict(
            eps_abs=1e-6,
            eps_rel=1e-6,
            alpha=1.6,
            max_iter=5000,
            adaptive_rho=True,
            adaptive_rho_interval=20,
            polish=True,
            check_termination=10,
            warm_start=True,
        )

    # ------------------------ control loop ------------------------
    def run(self):
        self.innit()
        time.sleep(0.2)
        dt = 1.0 / self.Hz
        self.control_stop = threading.Event()
        i = 0

        now0 = time.perf_counter()
        self._ctrl_start_wall = now0
        self._last_log_wall = now0

        # build QP once (structure + solver config)
        self.build_qp(dt=dt)

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                if i >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # ---- reference at this timestep ----
                p_ref = self.position_ref[i]
                v_ref = self.velocity_ref[i]
                R_ref = self.rotation_ref[i]
                w_ref = self.angular_velocity_ref[i]

                # ---- robot current state ----
                state = self.robot.get_state()
                p = np.array(state["gripper_world"][:3])
                v = np.array(state["speed_world"][:3])
                r = np.array(state["pose"][3:6])  # rotvec
                R = RR.from_rotvec(r).as_matrix()
                w = np.array(state["speed_world"][3:6])

                q    = self.robot.get_q()
                qdot = self.robot.get_qdot()

                J      = self.robot.get_J(q)
                Jdot   = self.robot.get_Jdot(q, qdot)
                M      = self.robot.get_M(q)
                Lambda = self.robot.get_Lambda(J, M)
                D      = self.robot.get_D(self.robot.K, Lambda)

                # ---- task errors ----
                e_p = p_ref - p           # position error
                e_v = v_ref - v           # linear vel error
                e_w = w_ref - w           # angular vel error
                e_R = R_ref @ R.T
                e_r = pin.log3(e_R)       # orientation error (rotvec)

                # ---- impedance -> desired wrench -> desired task accel ----
                f = self.robot.get_wrench_desired(D, self.robot.K, e_p, e_r, e_v, e_w)
                a = np.linalg.solve(Lambda, f)  # desired task-space accel (6,)

                # ---- feed params to QP ----
                self.J_p.value     = _freeze_sparsity(J)
                self.Jdot_p.value  = _freeze_sparsity(Jdot)
                self.qdot_p.value  = qdot
                self.a_p.value     = a
                self.q_p.value     = q

                # ---- solve QP ----
                _solve_t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                _solve_dt = time.perf_counter() - _solve_t0

                self._win_solver_time   += _solve_dt
                self._total_solver_time += _solve_dt

                # ---- extract solution ----
                qddot_sol = np.asarray(self.qddot_var.value).reshape(-1)

                # ---- diagnostics / objective breakdown ----
                Jp    = self.J_p.value
                Jdotp = self.Jdot_p.value
                ap    = self.a_p.value
                qdotp = self.qdot_p.value

                # e_acc_p = (J qddot + Jdot qdot - a)
                e_acc_p = Jp @ qddot_sol + Jdotp @ qdotp - ap   # shape (6,)
                W_diag  = self.W_task_vec_c.value              # [6,]

                obj_task_term = float(np.sum((W_diag * e_acc_p)**2))
                obj_reg_term  = float(self.lambda_reg) * float(qddot_sol @ qddot_sol)
                obj_total     = obj_task_term + obj_reg_term

                # split translation / rotation parts
                trans_err = e_acc_p[:3]
                rot_err   = e_acc_p[3:]
                w_lin_vec = W_diag[:3]
                w_rot_vec = W_diag[3:]
                obj_trans_term = float(np.sum((w_lin_vec * trans_err)**2))
                obj_rot_term   = float(np.sum((w_rot_vec * rot_err)**2))

                # ---- integrate solution for feedforward preview ----
                qdot_cmd = qdot + qddot_sol * dt
                q_cmd    = q    + qdot * dt + 0.5 * qddot_sol * dt * dt

                # ---- command robot ----
                self.robot.speedJ(qdot_cmd.tolist(), dt)

                # ---- log data ----
                tcp_log = dict(
                    p=p, v=v, rvec=r, w=w,
                    p_ref=p_ref, v_ref=v_ref,
                    rvec_ref=RR.from_matrix(R_ref).as_rotvec(),
                    w_ref=w_ref,
                    e_p=e_p, e_r=e_r, e_v=e_v, e_w=e_w
                )

                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": self.qp.status,
                    "obj": self.qp.value,

                    # joint space (measured + integrated-from-QP)
                    "q": q,
                    "qdot": qdot,
                    "qddot": qddot_sol,
                    "qdot_cmd": qdot_cmd,
                    "q_cmd": q_cmd,

                    # task space / impedance snapshot
                    "tcp": tcp_log,
                    "f": f,

                    # weights snapshot
                    "weights": {
                        "scale_lin": getattr(self, "scale_lin", None),
                        "scale_rot": getattr(self, "scale_rot", None),
                        "w_lin": getattr(self, "w_lin", None),
                        "w_rot": getattr(self, "w_rot", None),
                        "final_lin_weight": getattr(self, "final_lin_weight", None),
                        "final_rot_weight": getattr(self, "final_rot_weight", None),
                        "lambda_reg": getattr(self, "lambda_reg", None),
                    },

                    # objective breakdown for plotting
                    "obj_break": {
                        "trans": obj_trans_term,
                        "rot":   obj_rot_term,
                        "reg":   obj_reg_term,
                        "total": obj_total,
                    },
                })

            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.robot.speedStop()
                except Exception:
                    pass
                break

            # ---- realtime perf stats ----
            elapsed = time.perf_counter() - loop_start
            self._win_loop_time += elapsed
            self._win_iters += 1
            self._total_iters += 1
            if elapsed > dt + 1e-4:
                self._win_deadline_miss += 1
                self._total_deadline_miss += 1

            now = time.perf_counter()
            if now - self._last_log_wall >= self.log_every_s and self._win_iters > 0:
                avg_period     = self._win_loop_time / self._win_iters
                avg_hz         = (1.0 / avg_period) if avg_period > 0 else float('nan')
                avg_solver_ms  = (self._win_solver_time / self._win_iters) * 1000.0
                miss_pct       = (100.0 * self._win_deadline_miss / self._win_iters)
                print(f"[CTRL] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms/iter | "
                      f"deadline miss {miss_pct:5.1f}% over {self._win_iters} iters")
                self._win_iters = 0
                self._win_loop_time = 0.0
                self._win_solver_time = 0.0
                self._win_deadline_miss = 0
                self._last_log_wall = now

            time.sleep(max(0, dt - elapsed))
            i += 1

        # graceful shutdown
        try:
            self.robot.speedStop()
        except Exception:
            pass

        total_time = time.perf_counter() - self._ctrl_start_wall if self._ctrl_start_wall else 0.0
        overall_hz = (self._total_iters / total_time) if total_time > 0 else float('nan')
        overall_solver_ms = (self._total_solver_time / max(self._total_iters, 1)) * 1000.0
        overall_miss_pct  = (100.0 * self._total_deadline_miss / max(self._total_iters, 1))
        print(f"[CTRL SUMMARY] {overall_hz:6.2f} Hz over {self._total_iters} iters in {total_time:.2f}s | "
              f"solver avg {overall_solver_ms:.2f} ms | deadline miss {overall_miss_pct:.1f}%")

    # ------------------------ plotting ------------------------
    def plot_taskspace(self, title_prefix="TaskspaceTracking_SingleArm"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X", "Y", "Z"]

        key_tcp = "tcp"
        P        = np.array([d[key_tcp]["p"]        for d in self.control_data])
        V        = np.array([d[key_tcp]["v"]        for d in self.control_data])
        RVEC     = np.array([d[key_tcp]["rvec"]     for d in self.control_data])
        W        = np.array([d[key_tcp]["w"]        for d in self.control_data])
        P_ref    = np.array([d[key_tcp]["p_ref"]    for d in self.control_data])
        V_ref    = np.array([d[key_tcp]["v_ref"]    for d in self.control_data])
        RVEC_ref = np.array([d[key_tcp]["rvec_ref"] for d in self.control_data])
        W_ref    = np.array([d[key_tcp]["w_ref"]    for d in self.control_data])

        E_p = np.array([d[key_tcp]["e_p"] for d in self.control_data])
        E_v = np.array([d[key_tcp]["e_v"] for d in self.control_data])
        E_w = np.array([d[key_tcp]["e_w"] for d in self.control_data])
        E_r = np.array([d[key_tcp]["e_r"] for d in self.control_data])

        E_p_mag = np.linalg.norm(E_p, axis=1)
        E_v_mag = np.linalg.norm(E_v, axis=1)
        E_w_mag = np.linalg.norm(E_w, axis=1)
        E_r_mag = np.linalg.norm(E_r, axis=1)

        fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
        fig.suptitle(f"{title_prefix} – Robot R – {timestamp_str}", fontsize=14, y=0.99)

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
        fname = f"plots/{title_prefix.lower()}_R_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_jointspace(self, title_prefix="QP_and_Jointspace_SingleArm"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        status = [d.get("status", "") for d in self.control_data]

        Q         = np.vstack([d["q"]        for d in self.control_data])
        QDOT      = np.vstack([d["qdot"]     for d in self.control_data])
        QDDOT     = np.vstack([d["qddot"]    for d in self.control_data])
        QDOT_cmd  = np.vstack([d["qdot_cmd"] for d in self.control_data])
        Q_cmd     = np.vstack([d["q_cmd"]    for d in self.control_data])

        n = Q.shape[1]
        jlabels = [f"J{i+1}" for i in range(n)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        fig.suptitle(f"{title_prefix} – Robot R – {timestamp_str}", fontsize=14, y=0.98)

        ax = axes[0, 0]
        for j in range(n):
            ax.plot(t, Q[:, j], label=f"{jlabels[j]} meas")
            ax.plot(t, Q_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd(int)")
        ax.set_title("Joint Positions: measured vs integrated-from-QP")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(2*n, 6), fontsize=8)

        ax = axes[0, 1]
        for j in range(n):
            ax.plot(t, QDOT[:, j], label=f"{jlabels[j]} meas")
            ax.plot(t, QDOT_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd")
        ax.set_title("Joint Velocities: measured vs commanded")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(2*n, 6), fontsize=8)

        ax = axes[1, 0]
        for j in range(n):
            ax.plot(t, QDDOT[:, j], label=jlabels[j])
        ax.set_title("Joint Accelerations qddot (QP)")
        ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(n, 6), fontsize=8)

        def classify_status(s):
            s_lower = (s or "").lower()
            if s_lower == "optimal":
                return 0
            if s_lower in ("optimal_inaccurate", "user_limit", "max_iters_reached",
                           "iteration_limit_reached", "user_limit_reached"):
                return 1
            if any(key in s_lower for key in [
                "infeasible","unbounded","solver_error","error",
                "dual_infeasible","primal_infeasible"
            ]):
                return 2
            return 2

        class_vals = np.array([classify_status(s) for s in status], dtype=float)

        ax = axes[1, 1]
        ax.step(t, class_vals, where="post")
        ax.set_title("QP solve class (0=good,1=warn,2=bad)")
        ax.set_xlabel("time [s]")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["good","warn","bad"])
        ax.set_ylim(-0.5, 2.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_R_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        import datetime as dt

        os.makedirs("plots", exist_ok=True)
        ts_full = np.array([d["t"] for d in self.control_data])
        t_full  = ts_full - ts_full[0]

        mask = np.array([("obj_break" in d) for d in self.control_data], dtype=bool)
        if not np.any(mask):
            print("No obj_break logged; ensure you added the logging after solve.")
            return

        trans = np.array([d["obj_break"]["trans"] for d in self.control_data if "obj_break" in d])
        rot   = np.array([d["obj_break"]["rot"]   for d in self.control_data if "obj_break" in d])
        reg   = np.array([d["obj_break"]["reg"]   for d in self.control_data if "obj_break" in d])
        total = np.array([d["obj_break"]["total"] for d in self.control_data if "obj_break" in d])
        t_obj = t_full[mask]

        raw_status_list = [d.get("status", "") for d in self.control_data]

        def classify_status(s):
            s_lower = (s or "").lower()
            if s_lower == "optimal":
                return 0  # GOOD
            if s_lower in (
                "optimal_inaccurate",
                "user_limit",
                "max_iters_reached",
                "iteration_limit_reached",
                "user_limit_reached",
            ):
                return 1  # WARN
            if any(key in s_lower for key in [
                "infeasible",
                "unbounded",
                "solver_error",
                "error",
                "dual_infeasible",
                "primal_infeasible",
            ]):
                return 2  # BAD
            return 2

        class_vals = np.array([classify_status(s) for s in raw_status_list], dtype=float)

        unique_raw_statuses = []
        for s in raw_status_list:
            if s not in unique_raw_statuses:
                unique_raw_statuses.append(s)

        marker_cycle = ['o', 'x', '^', 's', 'D', 'v', 'P', '*']
        raw_status_meta = {}
        for idx, s in enumerate(unique_raw_statuses):
            raw_status_meta[s] = {
                "marker": marker_cycle[idx % len(marker_cycle)],
                "offset": (idx % 3) * 0.15 - 0.15
            }

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]}
        )

        fig.suptitle(
            f"{title_prefix} "
            f"(scale_lin={getattr(self,'scale_lin',None)}, "
            f"scale_rot={getattr(self,'scale_rot',None)}, "
            f"w_lin={getattr(self,'w_lin',None)}, "
            f"w_rot={getattr(self,'w_rot',None)}, "
            f"λ={getattr(self,'lambda_reg',None)})",
            fontsize=13, y=0.98
        )

        # subplot 1: objective breakdown
        ax1.plot(t_obj, total, label="total", linewidth=2)
        ax1.plot(t_obj, trans, label="translation term")
        ax1.plot(t_obj, rot,   label="rotation term")
        ax1.plot(t_obj, reg,   label="regularizer")
        ax1.set_ylabel("objective value")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=8)
        ax1.set_title("QP objective breakdown", fontsize=11)

        # subplot 2: solver status
        ax2.step(t_full, class_vals, where="post", linewidth=2, color="black", label="severity")

        ymin, ymax = -0.5, 2.5
        ax2.axhspan(-0.5, 0.5,  facecolor="green",  alpha=0.08)
        ax2.axhspan(0.5, 1.5,   facecolor="yellow", alpha=0.10)
        ax2.axhspan(1.5, 2.5,   facecolor="red",    alpha=0.08)

        legend_handles = []
        legend_labels  = []
        for raw_s in unique_raw_statuses:
            idxs = [i for i, ss in enumerate(raw_status_list) if ss == raw_s]
            if not idxs:
                continue

            xs = []
            ys = []
            for i_idx in idxs:
                xs.append(t_full[i_idx])
                base_y = class_vals[i_idx]
                jitter = raw_status_meta[raw_s]["offset"]
                ys.append(base_y + jitter)

            h = ax2.scatter(
                xs,
                ys,
                marker=raw_status_meta[raw_s]["marker"],
                s=40,
                linewidths=1,
                edgecolors="black",
                facecolors="none" if classify_status(raw_s) == 1 else "black",
                label=raw_s,
            )
            legend_handles.append(h)
            legend_labels.append(raw_s)

        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["GOOD", "WARN", "BAD"], fontsize=9)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel("time [s]")
        ax2.set_title("QP solve health + raw solver status", fontsize=11)
        ax2.grid(True, axis="x", alpha=0.3)

        if legend_handles:
            ax2.legend(legend_handles, legend_labels, loc="upper right",
                       fontsize=8, title="raw status")

        ts_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


class URImpedanceController(URForceController):
    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = K
        self.D = np.zeros((6, 6))

        # build Pinocchio model for kinematics/dynamics
        self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("tool0")

        # pre-cache state-ish
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
            self.pin_model,
            self.pin_data,
            self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(J)

    def get_Jdot(self, q, v):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model,
            self.pin_data,
            self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(dJ)

    def get_M(self, q):
        M = pin.crba(self.pin_model, self.pin_data, q)
        return np.asarray(0.5 * (M + M.T))

    def get_Lambda(self, J, M, lam2=1e-3):
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        A = 0.5 * (A + A.T)

        # damped "inverse" to get operational-space inertia
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_dinv = S / (S**2 + lam2)
        Lam = (Vt.T * S_dinv) @ U.T
        return 0.5 * (Lam + Lam.T)

    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda)
        S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        return np.real_if_close(0.5 * (D + D.T))

    def get_wrench_desired(self, D, K, e_p, e_r, e_v, e_w):
        Xd_err = np.hstack((e_v, e_w))
        X_err  = np.hstack((e_p, e_r))
        wrench = D @ Xd_err + K @ X_err
        return wrench


if __name__ == "__main__":
    # impedance gains: lower last 3 if orientation is too aggressive
    K = np.diag([400, 400, 400, 2, 2, 1])

    robotR = URImpedanceController(
        "192.168.1.33",
        K=K
    )

    trajectory = "motion_planner/trajectories/lift_100.npz"
    Hz = 90

    optimizer = JointOptimizationSingleArm(
        robot=robotR,
        Hz=Hz,
        trajectory=trajectory,
    )

    try:
        # user-defined approach positioning before control
        robotR.go_to_approach()
        robotR.wait_for_commands()
        robotR.wait_until_done()

        optimizer.run()

        optimizer.plot_taskspace()
        optimizer.plot_jointspace()
        optimizer.plot_qp_objective()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robotR.stop_control()
    except Exception as e:
        print(f"An error occurred: {e}")
        robotR.stop_control()
    finally:
        robotR.disconnect()
        print("Robot disconnected")
