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



class JointOptimization():
    def __init__(self, robotL, robotR, Hz, trajectory):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz

        self.trajectory_npz=np.load(trajectory)

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


        #limits
        self.joint_pose_limit = np.deg2rad(363)  # +- rad 360
        self.joint_speed_limit = np.deg2rad(190)  # +- rad/s 190
        self.joint_accel_limit = np.deg2rad(80)  # +- rad/s^2 80

        self.control_data = []

    def innit(self):
        #references
        self.traj_len = self.trajectory_npz['position'].shape[0]
        self.position_ref = self.trajectory_npz['position']
        self.velocity_ref = self.trajectory_npz['linear_velocity']
        self.rotation_ref = self.trajectory_npz['rotation_matrices']
        self.angular_velocity_ref = self.trajectory_npz['angular_velocity']

        self.position_ref_L = np.zeros_like(self.position_ref)
        self.position_ref_R = np.zeros_like(self.position_ref)
        self.rotation_ref_L = np.zeros_like(self.rotation_ref)
        self.rotation_ref_R = np.zeros_like(self.rotation_ref)

        start_position_L = np.array(self.robotL.get_state()["gripper_world"][:3])
        start_position_R = np.array(self.robotR.get_state()["gripper_world"][:3])
        start_rotation_L = np.array(RR.from_rotvec(self.robotL.get_state()["pose"][3:6]).as_matrix())
        start_rotation_R = np.array(RR.from_rotvec(self.robotR.get_state()["pose"][3:6]).as_matrix())
        for i in range(len(self.position_ref)):
            self.position_ref_L[i] = start_position_L + self.position_ref[i]
            self.position_ref_R[i] = start_position_R + self.position_ref[i]

        for j in range(len(self.rotation_ref)):
            self.rotation_ref_L[j] = start_rotation_L @ self.rotation_ref[j]
            self.rotation_ref_R[j] = start_rotation_R @ self.rotation_ref[j]

    def _freeze_sparsity(self, A, eps=1e-12):
        # Replace exact zeros with a tiny epsilon to keep nnz pattern constant.
        A = np.asarray(A, dtype=float).copy()
        A[A == 0.0] = eps
        return A

    def build_qp(self, dt):
        n_L = n_R = 6 
        # ---------- decision variables (once) ----------
        self.qddot_L_var = cp.Variable(n_L, name="qddot_L")
        self.qddot_R_var = cp.Variable(n_R, name="qddot_R")

        # ---------- parameters that will be updated every cycle ----------
        # Task-space dynamics
        self.J_L_p     = cp.Parameter((6, n_L), name="J_L")
        self.Jdot_L_p  = cp.Parameter((6, n_L), name="Jdot_L")
        self.qdot_L_p  = cp.Parameter(n_L, name="qdot_L")
        self.a_L_p     = cp.Parameter(6, name="a_L")  # a_L := (Λ_L^{-1} f_L)

        self.J_R_p     = cp.Parameter((6, n_R), name="J_R")
        self.Jdot_R_p  = cp.Parameter((6, n_R), name="Jdot_R")
        self.qdot_R_p  = cp.Parameter(n_R, name="qdot_R")
        self.a_R_p     = cp.Parameter(6, name="a_R")  # a_R := (Λ_R^{-1} f_R)

        # Joint-space kinematics for limits
        self.q_L_p     = cp.Parameter(n_L, name="q_L")
        self.q_R_p     = cp.Parameter(n_R, name="q_R")
        # If you ever change dt at runtime, make it a Parameter; else treat as constant
        self.dt_c      = cp.Constant(float(dt))
        self.dt2_c      = cp.Constant(float(dt*dt))

        # Limits (constants)
        q_pos_lim   = self.joint_pose_limit
        q_vel_lim   = self.joint_speed_limit
        q_acc_lim   = self.joint_accel_limit

        # objective
        xddot_L = self.J_L_p @ self.qddot_L_var + self.Jdot_L_p @ self.qdot_L_p
        xddot_R = self.J_R_p @ self.qddot_R_var + self.Jdot_R_p @ self.qdot_R_p
        e_L = xddot_L - self.a_L_p
        e_R = xddot_R - self.a_R_p
        obj = cp.sum_squares(e_L) + cp.sum_squares(e_R)

        # constraints (use constants dt_c, dt2_c)
        q_next_L    = self.q_L_p    + self.qdot_L_p * self.dt_c  + 0.5 * self.qddot_L_var * self.dt2_c
        q_next_R    = self.q_R_p    + self.qdot_R_p * self.dt_c  + 0.5 * self.qddot_R_var * self.dt2_c
        qdot_next_L = self.qdot_L_p + self.qddot_L_var * self.dt_c
        qdot_next_R = self.qdot_R_p + self.qddot_R_var * self.dt_c

        cons = [
            q_next_L    <=  q_pos_lim, q_next_L    >= -q_pos_lim,
            q_next_R    <=  q_pos_lim, q_next_R    >= -q_pos_lim,
            qdot_next_L <=  q_vel_lim, qdot_next_L >= -q_vel_lim,
            qdot_next_R <=  q_vel_lim, qdot_next_R >= -q_vel_lim,
            self.qddot_L_var <= q_acc_lim, self.qddot_L_var >= -q_acc_lim,
            self.qddot_R_var <= q_acc_lim, self.qddot_R_var >= -q_acc_lim,
        ]

        self.qp = cp.Problem(cp.Minimize(obj), cons)

        self.qp_kwargs= dict(
            eps_abs=3e-5,
            eps_rel=3e-5,
            max_iter=2000,
            adaptive_rho=True,
            adaptive_rho_interval=25,
            check_termination=10,
            # polish stays False
        )

    def run(self):
        #innit references
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

                # --- References ---
                p_ref_L = self.position_ref_L[i]
                p_ref_R = self.position_ref_R[i]
                v_ref_L  = self.velocity_ref[i]
                v_ref_R  = self.velocity_ref[i]
                R_ref_L = self.rotation_ref_L[i]
                R_ref_R = self.rotation_ref_R[i]
                w_ref_L = self.angular_velocity_ref[i]
                w_ref_R = self.angular_velocity_ref[i]

                # --- kinematics/dynamics LEFT ---
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

                f_L = self.robotL.get_wrench_desired(D_L, self.robotL.K, p_L, v_L, R_L, w_L, p_ref_L, v_ref_L, R_ref_L, w_ref_L)

                # --- kinematics/dynamics RIGHT ---
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

                f_R = self.robotR.get_wrench_desired(D_R, self.robotR.K, p_R, v_R, R_R, w_R, p_ref_R, v_ref_R, R_ref_R, w_ref_R)

                # --- QP Optimization ---
                a_L = np.linalg.solve(Lambda_L, f_L)
                a_R = np.linalg.solve(Lambda_R, f_R)

                # set Parameters
                # Densify to freeze sparsity
                J_L_d    = self._freeze_sparsity(J_L)
                Jdot_L_d = self._freeze_sparsity(Jdot_L)
                J_R_d    = self._freeze_sparsity(J_R)
                Jdot_R_d = self._freeze_sparsity(Jdot_R)

                self.J_L_p.value    = J_L_d
                self.Jdot_L_p.value = Jdot_L_d
                self.J_R_p.value    = J_R_d
                self.Jdot_R_p.value = Jdot_R_d
                
                self.qdot_L_p.value = qdot_L
                self.a_L_p.value    = a_L

                self.qdot_R_p.value = qdot_R
                self.a_R_p.value    = a_R

                self.q_L_p.value    = q_L
                self.q_R_p.value    = q_R

                # solve QP
                _solve_t0 = time.perf_counter()
                try:
                    self.qp.solve(**self.qp_kwargs)
                except Exception as e:
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

                # integrate
                qdot_L_cmd = (qdot_L + qddot_L_sol*dt)
                q_L_cmd    = q_L    + qdot_L*dt + 0.5*qddot_L_sol*dt*dt

                qdot_R_cmd = (qdot_R + qddot_R_sol*dt)
                q_R_cmd    = q_R    + qdot_R*dt + 0.5*qddot_R_sol*dt*dt

                # send commands
                self.robotL.speedJ(qdot_L_cmd.tolist(), dt)
                self.robotR.speedJ(qdot_R_cmd.tolist(), dt)


                # measured task states (use what you already computed)
                tcp_log_L = dict(
                    p=p_L, v=v_L, rvec=r_L, w=w_L,
                    p_ref=p_ref_L, v_ref=v_ref_L,
                    rvec_ref=RR.from_matrix(R_ref_L).as_rotvec(),
                    w_ref=w_ref_L
                )
                tcp_log_R = dict(
                    p=p_R, v=v_R, rvec=r_R, w=w_R,
                    p_ref=p_ref_R, v_ref=v_ref_R,
                    rvec_ref=RR.from_matrix(R_ref_R).as_rotvec(),
                    w_ref=w_ref_R
                )

                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": self.qp.status,
                    "obj": self.qp.value,
                    # joint space
                    "q_L": q_L, "qdot_L": qdot_L, "qddot_L": qddot_L_sol, "qdot_cmd_L": qdot_L_cmd,
                    "q_R": q_R, "qdot_R": qdot_R, "qddot_R": qddot_R_sol, "qdot_cmd_R": qdot_R_cmd,
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

            elapsed = time.perf_counter() - loop_start
            self._win_loop_time += elapsed
            self._win_iters += 1
            self._total_iters += 1
            if elapsed > dt + 1e-4:
                self._win_deadline_miss += 1
                self._total_deadline_miss += 1

            # print every log_every_s seconds (not every cycle)
            now = time.perf_counter()
            if now - self._last_log_wall >= self.log_every_s and self._win_iters > 0:
                avg_period = self._win_loop_time / self._win_iters
                avg_hz = (1.0 / avg_period) if avg_period > 0 else float('nan')
                avg_solver_ms = (self._win_solver_time / self._win_iters) * 1000.0
                miss_pct = (100.0 * self._win_deadline_miss / self._win_iters)

                # single concise line
                print(f"[CTRL] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms/iter | deadline miss {miss_pct:5.1f}% "
                      f"over {self._win_iters} iters")

                # reset window
                self._win_iters = 0
                self._win_loop_time = 0.0
                self._win_solver_time = 0.0
                self._win_deadline_miss = 0
                self._last_log_wall = now

            # Keep loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, dt - elapsed))
            i += 1

        # Clean stop
        try:
            self.robotL.speedStop()
            self.robotR.speedStop()
        except:
            pass

        total_time = time.perf_counter() - self._ctrl_start_wall if self._ctrl_start_wall else 0.0
        overall_hz = (self._total_iters / total_time) if total_time > 0 else float('nan')
        overall_solver_ms = (self._total_solver_time / max(self._total_iters,1)) * 1000.0
        overall_miss_pct = (100.0 * self._total_deadline_miss / max(self._total_iters,1))
        print(f"[CTRL SUMMARY] {overall_hz:6.2f} Hz over {self._total_iters} iters in {total_time:.2f}s | "
              f"solver avg {overall_solver_ms:.2f} ms | deadline miss {overall_miss_pct:.1f}%")

    def plot_taskspace_tracking(self, robots=("L", "R"), title_prefix="TaskspaceTracking"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # define color map (same order for all)
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X", "Y", "Z"]

        for side in robots:
            key_tcp = f"tcp_{side}"
            if key_tcp not in self.control_data[0]:
                print(f"Missing {key_tcp}; skipping {side}.")
                continue

            P      = np.array([d[key_tcp]["p"]        for d in self.control_data])
            V      = np.array([d[key_tcp]["v"]        for d in self.control_data])
            RVEC   = np.array([d[key_tcp]["rvec"]     for d in self.control_data])
            W      = np.array([d[key_tcp]["w"]        for d in self.control_data])
            P_ref    = np.array([d[key_tcp]["p_ref"]    for d in self.control_data])
            V_ref    = np.array([d[key_tcp]["v_ref"]    for d in self.control_data])
            RVEC_ref = np.array([d[key_tcp]["rvec_ref"] for d in self.control_data])
            W_ref    = np.array([d[key_tcp]["w_ref"]    for d in self.control_data])

            fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
            fig.suptitle(f"{title_prefix} – Robot {side} – {timestamp_str}", fontsize=14, y=0.98)

            # ---------------- Position ----------------
            ax = axes[0, 0]
            for i in range(3):
                ax.plot(t, P[:, i], color=colors[i], label=f"p{labels[i]}")
                ax.plot(t, P_ref[:, i], "--", color=colors[i], alpha=0.8, label=f"p{labels[i]} ref")
            ax.set_title("Position vs Ref")
            ax.set_ylabel("m")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            # ---------------- Rotation (rotvec) ----------------
            ax = axes[0, 1]
            for i in range(3):
                ax.plot(t, RVEC[:, i], color=colors[i], label=f"r{labels[i]}")
                ax.plot(t, RVEC_ref[:, i], "--", color=colors[i], alpha=0.8, label=f"r{labels[i]} ref")
            ax.set_title("Rotation (rotvec) vs Ref")
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            # ---------------- Linear velocity ----------------
            ax = axes[1, 0]
            for i in range(3):
                ax.plot(t, V[:, i], color=colors[i], label=f"v{labels[i]}")
                ax.plot(t, V_ref[:, i], "--", color=colors[i], alpha=0.8, label=f"v{labels[i]} ref")
            ax.set_title("Linear Velocity vs Ref")
            ax.set_xlabel("s")
            ax.set_ylabel("m/s")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            # ---------------- Angular velocity ----------------
            ax = axes[1, 1]
            for i in range(3):
                ax.plot(t, W[:, i], color=colors[i], label=f"w{labels[i]}")
                ax.plot(t, W_ref[:, i], "--", color=colors[i], alpha=0.8, label=f"w{labels[i]} ref")
            ax.set_title("Angular Velocity vs Ref")
            ax.set_xlabel("s")
            ax.set_ylabel("rad/s")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"plots/{title_prefix.lower()}_{side}_{timestamp_str}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {fname}")


    def plot_qp_and_jointspace(self, robots=("L", "R"), title_prefix="QP_and_Jointspace"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # QP objective (single figure)
        obj = np.array([d.get("obj", np.nan) for d in self.control_data])
        status = [d.get("status", "") for d in self.control_data]

        plt.figure(figsize=(12, 4))
        plt.plot(t, obj, label="QP objective")
        bad = np.array([i for i, s in enumerate(status) if s not in ("optimal", "optimal_inaccurate")], dtype=int)
        if bad.size:
            plt.scatter(t[bad], obj[bad], marker="x", s=30, label="non-optimal", zorder=3)
        plt.title(f"QP Objective over Time – {timestamp_str}")
        plt.xlabel("time [s]"); plt.ylabel("objective"); plt.grid(True, alpha=0.3); plt.legend()
        fname_obj = f"plots/{title_prefix.lower()}_objective_{timestamp_str}.png"
        plt.tight_layout(); plt.savefig(fname_obj, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname_obj}")

        # Per-robot joint figures
        for side in robots:
            sample = self.control_data[0]
            q_key, qdot_key, qddot_key, qdot_cmdkey = f"q_{side}", f"qdot_{side}", f"qddot_{side}", f"qdot_cmd_{side}"
            missing = [k for k in (q_key, qdot_key, qddot_key, qdot_cmdkey) if k not in sample]
            if missing:
                print(f"Missing {missing} for {side}; skipping.")
                continue

            Q       = np.vstack([d[q_key]        for d in self.control_data])  # (N,n)
            QDOT    = np.vstack([d[qdot_key]     for d in self.control_data])
            QDDOT   = np.vstack([d[qddot_key]    for d in self.control_data])
            QDOTcmd = np.vstack([d[qdot_cmdkey]  for d in self.control_data])

            n = Q.shape[1]
            jlabels = [f"J{i+1}" for i in range(n)]

            fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
            fig.suptitle(f"{title_prefix} – Robot {side} – {timestamp_str}", fontsize=14, y=0.98)

            # q
            ax = axes[0, 0]
            for j in range(n):
                ax.plot(t, Q[:, j], label=jlabels[j])
            ax.set_title("Joint Positions q"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3); ax.legend(ncol=min(n, 6), fontsize=8)

            # qdot vs qdot_cmd
            ax = axes[0, 1]
            for j in range(n):
                ax.plot(t, QDOT[:, j], label=f"{jlabels[j]} meas")
                ax.plot(t, QDOTcmd[:, j], "--", alpha=0.8, label=f"{jlabels[j]} cmd")
            ax.set_title("Joint Velocities: measured vs commanded"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=min(2*n, 6), fontsize=8)

            # qddot (QP)
            ax = axes[1, 0]
            for j in range(n):
                ax.plot(t, QDDOT[:, j], label=jlabels[j])
            ax.set_title("Joint Accelerations qddot (QP)"); ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²"); ax.grid(True, alpha=0.3); ax.legend(ncol=min(n, 6), fontsize=8)

            # feasibility flag
            ax = axes[1, 1]
            y = np.array([0 if s in ("optimal", "optimal_inaccurate") else 1 for s in status], dtype=float)
            ax.step(t, y, where="post")
            ax.set_title("QP infeasible/non-optimal"); ax.set_xlabel("time [s]"); ax.set_yticks([0, 1]); ax.set_yticklabels(["ok", "bad"]); ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"plots/{title_prefix.lower()}_{side}_{timestamp_str}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved: {fname}")


class URImpedanceController(URForceController):
    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = K
        self.D = np.zeros((6, 6))

        self.pin_model = pin.buildModelFromUrdf(
            "ur5/ur5e.urdf"
        )
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("wrist_3_link")

        self.q = self.get_q()
        self.qdot = self.get_qdot()
        self.J = self.get_J(self.q)
        self.Jdot =  self.get_Jdot(self.q, self.qdot)
        self.M = self.get_M(self.q)
        self.Lambda= self.get_Lambda(self.J, self.M)
        self.D = self.get_D(self.K, self.Lambda)

        self.cotrol_data = []


    def get_q(self):
        q = np.array(self.rtde_receive.getActualQ(), dtype=float)
        return q
    
    def get_qdot(self):
        v = np.array(self.rtde_receive.getActualQd(), dtype=float)  # joint velocities
        return v

    def get_J(self,q):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.pin_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(J)
    
    def get_Jdot(self, q, v):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model, self.pin_data, self.pin_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(dJ)

    def get_M(self, q):
        M = pin.crba(self.pin_model, self.pin_data, q)
        M = (M + M.T) / 2.0
        return np.asarray(M)
    
    def get_Lambda(self, J, M, eps=1e-6):
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        A += eps * np.eye(6)
        return np.linalg.inv(A)

    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda); S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        D = np.real_if_close(D)
        return 0.5*(D + D.T)
    
    def get_wrench_desired(self, D, K, p, v, R, w, p_ref, v_ref, R_ref, w_ref):
        # velocity error
        Xd_err = np.hstack((v_ref - v, w_ref - w))

        # position error
        P_err = p_ref - p
        # rotation error
        R_err = R_ref @ R.T #try log later
        r_err = RR.from_matrix(R_err).as_rotvec() 

        X_err = np.hstack((P_err, r_err))

        wrench = D @ Xd_err + K @ X_err

        return wrench