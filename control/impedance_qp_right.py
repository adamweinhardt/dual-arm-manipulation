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

from utils.utils import _freeze_sparsity


class JointOptimizationSingleArm():
    def __init__(self, robot, Hz, trajectory):
        self.robot = robot
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
        Load trajectory directly in WORLD by treating `position` as deltas from the
        current TCP and keeping the initial gripper orientation. No box frame.
        """
        # trajectory
        self.traj_len = self.trajectory_npz['position'].shape[0]
        p_deltas  = self.trajectory_npz['position']            # (T,3), treated as WORLD deltas
        v_world   = self.trajectory_npz['linear_velocity']     # (T,3), WORLD linear v

        # initial WORLD TCP pose/orientation
        state = self.robot.get_state()
        p_gripper_init = np.array(state["gripper_world"][:3])
        R_gripper_init = RR.from_rotvec(state["pose"][3:6]).as_matrix()

        # allocate WORLD references
        self.position_ref         = np.zeros_like(p_deltas)
        self.velocity_ref         = np.zeros_like(v_world)
        self.rotation_ref         = np.zeros((self.traj_len, 3, 3))
        self.angular_velocity_ref = np.zeros_like(p_deltas)  # zeros; rotation disabled anyway

        for t in range(self.traj_len):
            self.position_ref[t] = p_gripper_init + p_deltas[t]
            self.velocity_ref[t] = v_world[t]
            self.rotation_ref[t] = R_gripper_init  # keep constant

    # ------------------------ QP build ------------------------
    def build_qp(self, dt):
        n = 6
        self.qddot_var = cp.Variable(n, name="qddot")

        # parameters (update each cycle)
        self.J_p     = cp.Parameter((6, n), name="J")
        self.Jdot_p  = cp.Parameter((6, n), name="Jdot")
        self.qdot_p  = cp.Parameter(n, name="qdot")
        self.a_p     = cp.Parameter(6, name="a")
        self.q_p     = cp.Parameter(n, name="q")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(dt * dt))

        # objective: accel tracking
        xddot = self.J_p @ self.qddot_var + self.Jdot_p @ self.qdot_p
        e = xddot - self.a_p
        if self.disable_rotation:
            W6 = cp.Constant(np.diag([1, 1, 1, 0, 0, 0]))
            obj = cp.sum_squares(W6 @ e)
        else:
            obj = cp.sum_squares(e)

        # joint constraints
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

        if self.disable_rotation:
            # Enforce zero angular task acceleration (rows 3:6 are angular)
            cons += [
                xddot[3:6] == 0
            ]


        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=16000,
            adaptive_rho=True,
            adaptive_rho_interval=40,
            polish=False,
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

        self.build_qp(dt=dt)

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                if i >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # ---- references (WORLD) ----
                p_ref = self.position_ref[i]
                v_ref = self.velocity_ref[i]
                R_ref = self.rotation_ref[i]
                w_ref = self.angular_velocity_ref[i]  # zeros

                # ---- Kinematics/dynamics (WORLD) ----
                state = self.robot.get_state()
                p = np.array(state["gripper_world"][:3])
                v = np.array(state["speed_world"][:3])
                r = np.array(state["pose"][3:6])  # rotvec
                R = RR.from_rotvec(r).as_matrix()
                w = np.array(state["speed_world"][3:6])

                q = self.robot.get_q()
                qdot = self.robot.get_qdot()

                J = self.robot.get_J(q)
                Jdot = self.robot.get_Jdot(q, qdot)
                M = self.robot.get_M(q)
                Lambda = self.robot.get_Lambda(J, M)
                D = self.robot.get_D(self.robot.K, Lambda)

                # errors
                e_p = p_ref - p
                e_v = v_ref - v
                e_w = w_ref - w
                e_R = R_ref @ R.T
                e_r = RR.from_matrix(e_R).as_rotvec()

                if self.disable_rotation:
                    e_r[:] = 0.0
                    e_w[:] = 0.0
                    D[3:, :] = 0.0
                    D[:, 3:] = 0.0

                f = self.robot.get_wrench_desired(D, self.robot.K, e_p, e_r, e_v, e_w)

                # desired task accelerations (WORLD) for QP
                a = np.linalg.solve(Lambda, f)
                if self.disable_rotation:
                    a[3:] = 0.0

                # set Parameters (freeze sparsity)
                self.J_p.value    = _freeze_sparsity(J)
                self.Jdot_p.value = _freeze_sparsity(Jdot)
                self.qdot_p.value = qdot
                self.a_p.value    = a
                self.q_p.value    = q

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
                    qddot_sol = np.zeros_like(q)
                else:
                    qddot_sol = np.asarray(self.qddot_var.value).reshape(-1)

                # diagnostics
                if i % 50 == 0:
                    pass

                # integrate solver outputs (for plotting & feed)
                qdot_cmd = qdot + qddot_sol * dt
                q_cmd    = q    + qdot * dt + 0.5 * qddot_sol * dt * dt

                # send commands
                self.robot.speedJ(qdot_cmd.tolist(), dt)

                # log
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
                    "q": q, "qdot": qdot, "qddot": qddot_sol,
                    "qdot_cmd": qdot_cmd, "q_cmd": q_cmd,
                    # task space
                    "tcp": tcp_log,
                    # impedance bits
                    "f": f,
                })

            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.robot.speedStop()
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
            self.robot.speedStop()
        except:
            pass

        total_time = time.perf_counter() - self._ctrl_start_wall if self._ctrl_start_wall else 0.0
        overall_hz = (self._total_iters / total_time) if total_time > 0 else float('nan')
        overall_solver_ms = (self._total_solver_time / max(self._total_iters, 1)) * 1000.0
        overall_miss_pct = (100.0 * self._total_deadline_miss / max(self._total_iters, 1))
        print(f"[CTRL SUMMARY] {overall_hz:6.2f} Hz over {self._total_iters} iters in {total_time:.2f}s | "
              f"solver avg {overall_solver_ms:.2f} ms | deadline miss {overall_miss_pct:.1f}%")

    # ------------------------ plotting ------------------------
    def plot_taskspace_tracking(self, title_prefix="TaskspaceTracking_SingleArm"):
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

    def plot_qp_and_jointspace(self, title_prefix="QP_and_Jointspace_SingleArm"):
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

        ax = axes[1, 1]
        y = np.array([0 if s in ("optimal", "optimal_inaccurate") else 1 for s in status], dtype=float)
        ax.step(t, y, where="post")
        ax.set_title("QP infeasible/non-optimal")
        ax.set_xlabel("time [s]")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["ok", "bad"])
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_R_{timestamp_str}.png"
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

if __name__ == "__main__":
    # Impedance gains (same pattern you used)
    K = np.diag([1500, 1500, 1500, 100, 100, 100])

    # ---- RIGHT robot only ----
    robotR = URImpedanceController(
        "192.168.1.33", K=K
    )

    # Trajectory (BOX frame); adjust path if needed
    trajectory = "motion_planner/trajectories/lifting.npz"
    Hz = 70

    # Create right-arm optimizer; start with rotation disabled to simplify debugging
    optimizer = JointOptimizationSingleArm(
        robot=robotR,
        Hz=Hz,
        trajectory=trajectory,  # set False later when linear part looks good
    )

    try:
        # --- Bring the right arm to a known state (adapt these to your setup) ---
        robotR.go_home()
        robotR.wait_for_commands()
        robotR.wait_until_done()

        robotR.go_to_approach()
        robotR.wait_for_commands()
        robotR.wait_until_done()

        optimizer.run()

        # --- Plots (RIGHT only) ---
        optimizer.plot_taskspace_tracking(title_prefix="TaskspaceTracking_R")
        optimizer.plot_qp_and_jointspace(title_prefix="QP_and_Jointspace_R")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robotR.stop_control()
    except Exception as e:
        print(f"An error occurred: {e}")
        robotR.stop_control()
    finally:
        robotR.disconnect()
        print("Robot disconnected")