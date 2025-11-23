#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import threading
import numpy as np
import cvxpy as cp
import pinocchio as pin
from scipy.spatial.transform import Rotation as RR
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from control.PID.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity, _as_rowvec_1d


# ===========================
# helpers
# ===========================

def diag6(vals):
    """
    Build a 6x6 diagonal matrix from 1, 3, or 6 values:
      - 1 value v   -> diag([v]*6)
      - 3 values a  -> diag([ax, ay, az, ax, ay, az])
      - 6 values    -> diag(vals)
    """
    a = np.asarray(vals, dtype=float).reshape(-1)
    if a.size == 1:
        return np.eye(6) * a.item()
    if a.size == 3:
        return np.diag([a[0], a[1], a[2], a[0], a[1], a[2]])
    if a.size == 6:
        return np.diag(a)
    raise ValueError("diag6 expects 1, 3, or 6 values")


def is_finite(*arrays):
    return all(np.all(np.isfinite(np.asarray(x))) for x in arrays)

def short_arc_log(R_ref, R_cur):
    # quats: [x, y, z, w] from scipy
    q_ref = RR.from_matrix(R_ref).as_quat()
    q_cur = RR.from_matrix(R_cur).as_quat()
    # ensure shortest arc
    if np.dot(q_ref, q_cur) < 0.0:
        q_ref = -q_ref
    R_rel = (RR.from_quat(q_ref) * RR.from_quat(q_cur).inv()).as_matrix()
    return pin.log3(R_rel)


# ===========================
# robot wrapper
# ===========================

class URImpedanceController(URForceController):
    """
    Thin wrapper to expose kinematics/dynamics via Pinocchio.
    K is the impedance stiffness (6x6, SPD).
    """

    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = np.asarray(K, dtype=float)
        self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("tool0")

    # state
    def get_q(self):
        return np.array(self.rtde_receive.getActualQ(), dtype=float)

    def get_qdot(self):
        return np.array(self.rtde_receive.getActualQd(), dtype=float)

    # kinematics
    def get_J(self, q):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(J)

    def get_Jdot(self, q, v):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model, self.pin_data, self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return self.get_J_world(dJ)

    # dynamics
    def get_M(self, q):
        M = pin.crba(self.pin_model, self.pin_data, q)
        return np.asarray(0.5 * (M + M.T))

    def get_Lambda(self, J, M, tikhonov=1e-6):
        """
        TRUE operational-space inertia:
            Λ ≈ (J M^{-1} Jᵀ + λI)^{-1}
        """
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        w_d = np.maximum(w + float(tikhonov), 1e-12)
        Lambda = V @ np.diag(1.0 / w_d) @ V.T
        return np.asarray(0.5 * (Lambda + Lambda.T))

    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda)
        S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        return np.real_if_close(0.5 * (D + D.T))

    def wrench_desired(self, K, D, e_p, e_r, e_v, e_w):
        X_err  = np.hstack((e_p, e_r))
        Xd_err = np.hstack((e_v, e_w))
        return D @ Xd_err + K @ X_err


# ===========================
# single-arm QP controller
# ===========================

class SingleArmImpedanceQP:
    """
    Each tick, solve:
        min  || W_imp (J qdd + Jdot qdot - a_des) ||^2 + λ_reg ||qdd||^2
        s.t. q_next    = q + qdot dt + 0.5 qdd dt^2
             qdot_next = qdot + qdd dt
             -q_pos_lim ≤ q_next    ≤ q_pos_lim
             -q_vel_lim ≤ qdot_next ≤ q_vel_lim
             -q_acc_lim ≤ qdd       ≤ q_acc_lim
    """

    def __init__(
        self,
        robot: URImpedanceController,
        Hz: int,
        trajectory_npz_path: str,
        W_imp=None,
        lambda_reg: float = 1e-2,
        tikhonov_lambda: float = 1e-6,   # for Λ
    ):
        self.robot = robot
        self.Hz = int(Hz)
        self.dt = 1.0 / float(self.Hz)

        # refs
        self._npz_path = trajectory_npz_path

        # symmetric limits
        self.joint_pose_limit  = np.deg2rad(360.0)   # rad
        self.joint_speed_limit = np.deg2rad(180.0)   # rad/s
        self.joint_accel_limit = np.deg2rad(5000)   # rad/s^2

        # weights
        self.W_imp_np = np.asarray(W_imp, dtype=float)
        self.lambda_reg = float(lambda_reg)
        self.tikhonov_lambda = float(tikhonov_lambda)

        # buffers / stats
        self.control_data = []
        self._should_stop = threading.Event()

        # QP
        self._build_qp(self.dt)

    # ----- references -----
    def _load_refs(self):
            data = np.load(self._npz_path)
            p_box = data["position"]
            v_box = data["linear_velocity"]
            R_box = data["rotation_matrices"]
            w_box = data["angular_velocity"]
            self.traj_len = int(p_box.shape[0])

            # --- current robot state ---
            state = self.robot.get_state()
            # Keep p0, R0 as the physical starting pose of THIS specific robot
            p0 = np.array(state["gripper_world"][:3])                   
            R0 = RR.from_rotvec(state["pose"][3:6]).as_matrix()         

            # --- preallocate ---
            self.p_ref = np.zeros_like(p_box)
            self.v_ref = np.zeros_like(v_box)
            self.R_ref = np.zeros_like(R_box)
            self.w_ref = np.zeros_like(w_box)

            # --- baseline (box start) ---
            R_box0 = R_box[0]

            for t in range(self.traj_len):
                # --- translational refs ---
                self.p_ref[t] = p0 + p_box[t]
                self.v_ref[t] = v_box[t]

                # --- relative box rotation ---
                R_rel = R_box[t] @ R_box0.T

                self.R_ref[t] = R_rel @ R0  
                
                self.w_ref[t] = w_box[t]

    # ----- QP model -----
    def _build_qp(self, dt):
        n = 6
        self.qdd = cp.Variable(n, name="qdd")

        # parameters
        self.J_p     = cp.Parameter((6, n), name="J")
        self.Jdot_p  = cp.Parameter((6, n), name="Jdot")
        self.q_p     = cp.Parameter(n,      name="q")
        self.qdot_p  = cp.Parameter(n,      name="qdot")
        self.a_des_p = cp.Parameter(6,      name="a_des")
        self.W_imp_p = cp.Parameter((6, 6), name="W_imp")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(0.5 * dt * dt))

        # one-step predictions
        q_next    = self.q_p    + self.qdot_p * self.dt_c + self.qdd * self.dt2_c
        qdot_next = self.qdot_p + self.qdd * self.dt_c

        # residual
        e_imp = self.J_p @ self.qdd + self.Jdot_p @ self.qdot_p - self.a_des_p

        # objective
        obj = cp.sum_squares(self.W_imp_p @ e_imp) + self.lambda_reg * cp.sum_squares(self.qdd)

        # hard constraints
        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_acc_lim = self.joint_accel_limit
        cons = [
            -q_pos_lim <= q_next,    q_next    <= q_pos_lim,
            -q_vel_lim <= qdot_next, qdot_next <= q_vel_lim,
            -q_acc_lim <= self.qdd,  self.qdd   <= q_acc_lim,
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.prob_kwargs = dict(
            eps_abs=1e-7,
            eps_rel=1e-7,
            alpha=1.6,
            max_iter=15000,
            adaptive_rho=True,
            adaptive_rho_interval=45,
            polish=True,
            check_termination=10,
            warm_start=True,
        )

    # ----- control loop -----
    def run(self):
        self._load_refs()
        time.sleep(0.1)

        dt = self.dt
        t_idx = 0
        i = 0

        self._should_stop.clear()
        ctrl_t0 = time.perf_counter()
        last_log = ctrl_t0

        win_iters = 0
        win_loop_t = 0.0
        win_solver_t = 0.0
        win_deadline_miss = 0

        total_iters = 0
        total_solver_t = 0.0
        total_deadline_miss = 0

        print("[SingleArmQP] Starting control loop (hard bounds)")

        while not self._should_stop.is_set():
            loop_t0 = time.perf_counter()
            try:
                if t_idx >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # refs
                p_ref = self.p_ref[t_idx]
                v_ref = self.v_ref[t_idx]
                R_ref = self.R_ref[t_idx]
                w_ref = self.w_ref[t_idx]

                # state
                state = self.robot.get_state()
                p = np.array(state["gripper_world"][:3])
                v = np.array(state["speed_world"][:3])
                rotvec = np.array(state["pose"][3:6])
                R = RR.from_rotvec(rotvec).as_matrix()
                w = np.array(state["speed_world"][3:6])

                q    = self.robot.get_q()
                qdot = self.robot.get_qdot()

                # model
                J    = self.robot.get_J(q)
                Jdot = self.robot.get_Jdot(q, qdot)
                M    = self.robot.get_M(q)
                Lam  = self.robot.get_Lambda(J, M, tikhonov=self.tikhonov_lambda)  # TRUE Λ
                D    = self.robot.get_D(self.robot.K, Lam)

                # errors
                e_p = p_ref - p
                e_v = v_ref - v
                e_R = R_ref @ R.T
                #e_r = pin.log3(e_R)
                e_r = short_arc_log(R_ref, R)
                e_w = w_ref - w

                # desired wrench/accel (ẍ = Λ^{-1} F)
                f_des = self.robot.wrench_desired(self.robot.K, D, e_p, e_r, e_v, e_w)
                a_des = np.linalg.solve(Lam, f_des)

                if not is_finite(J, Jdot, q, qdot, a_des):
                    raise ValueError("Non-finite detected in inputs to QP")

                # feed
                self.J_p.value = _freeze_sparsity(J)
                self.Jdot_p.value = _freeze_sparsity(Jdot)
                self.q_p.value = q
                self.qdot_p.value = qdot
                self.a_des_p.value = a_des
                self.W_imp_p.value = self.W_imp_np

                # solve
                t0 = time.perf_counter()
                self.prob.solve(solver=cp.OSQP, **self.prob_kwargs)
                solve_dt = time.perf_counter() - t0
                win_solver_t += solve_dt
                total_solver_t += solve_dt

                status = (self.prob.status or "").lower()
                ok = status in ("optimal", "optimal_inaccurate")
                qdd_sol = _as_rowvec_1d(self.qdd.value, "qdd", length=6) if ok and self.qdd.value is not None else np.zeros(6)

                # command
                qdot_cmd = qdot + qdd_sol * dt
                self.robot.speedJ(qdot_cmd.tolist(), dt)

                # log objective decomposition
                e_imp_val = J @ qdd_sol + Jdot @ qdot - a_des
                imp_term = float((self.W_imp_np @ e_imp_val).T @ (self.W_imp_np @ e_imp_val))
                reg_term = float(self.lambda_reg) * float(qdd_sol @ qdd_sol)
                obj_total = imp_term + reg_term

                # --- DEBUG PULSE ------------------------------------------------------------
                if i % 50 == 0:
                    # keep a previous orientation for finite-diff angular velocity
                    if not hasattr(self, "_dbg_prev_R"):
                        self._dbg_prev_R = R
                        self._dbg_prev_r = RR.from_matrix(R).as_rotvec()

                    arm = "LEFT" if str(getattr(self.robot, "ip", "")).endswith(".33") else ("RIGHT" if str(getattr(self.robot,"ip","")).endswith(".66") else "?")

                    # frames:
                    #   J : LOCAL_WORLD_ALIGNED  (WORLD axes)
                    #   w_meas_W : WORLD (from state["speed_world"])
                    #   w_meas_BASE : ROBOT BASE (from state["speed"])
                    #   w_pred_W : WORLD (J q̇)
                    xdot_pred = J @ qdot
                    w_pred_W  = xdot_pred[3:]

                    w_meas_W     = np.array(state["speed_world"][3:6])
                    w_meas_BASE  = np.array(state["speed"][3:6])

                    # orientation errors (in WORLD vs BODY conventions)
                    r_left  = pin.log3(R_ref @ R.T)      # left-invariant (WORLD axes)
                    r_right = pin.log3(R.T @ R_ref)      # right-invariant (BODY axes)

                    # finite-diff angular velocity in WORLD
                    R_prev  = self._dbg_prev_R
                    r_prev  = self._dbg_prev_r
                    w_fd1_W = (RR.from_matrix(R).as_rotvec() - r_prev) / max(dt, 1e-6)
                    dR      = R @ R_prev.T
                    w_fd2_W = (R @ pin.log3(dR)) / max(dt, 1e-6)

                    # Λ conditioning
                    ew = np.linalg.eigvalsh(0.5 * (Lam + Lam.T))
                    lam_min = float(max(np.min(ew), 1e-12))
                    lam_max = float(max(np.max(ew), lam_min))
                    condL   = lam_max / lam_min

                    print(f"\n[DBG {arm}] t={i*dt:5.2f}s  (J: WORLD axes | ω_meas: WORLD & BASE | ω_pred: Jq̇ WORLD)")
                    print(f"  |r|_left  (WORLD) = {np.linalg.norm(r_left):.3f}   r_left  = {np.round(r_left, 3)}")
                    print(f"  |r|_right (BODY)  = {np.linalg.norm(r_right):.3f}  r_right = {np.round(r_right, 3)}")
                    print(f"  ω_meas_W         = {np.round(w_meas_W, 3)}")
                    print(f"  ω_pred_Jq̇_W     = {np.round(w_pred_W, 3)}   ‖Δ‖ = {np.linalg.norm(w_meas_W - w_pred_W):.3e}")
                    print(f"  ω_meas_BASE      = {np.round(w_meas_BASE, 3)}   (UR base frame)")
                    print(f"  ω_fd1_W(rotvec)  = {np.round(w_fd1_W, 3)}")
                    print(f"  ω_fd2_W(logmap)  = {np.round(w_fd2_W, 3)}")
                    print(f"  Λ eig[min,max]   = [{lam_min:.3e}, {lam_max:.3e}]   cond(Λ) = {condL:.2e}")

                    # update prevs
                    self._dbg_prev_R = R
                    self._dbg_prev_r = RR.from_matrix(R).as_rotvec()
                # ---------------------------------------------------------------------------


                self.control_data.append({
                    "t": time.time(), "i": i, "status": self.prob.status, "obj": obj_total,
                    "q": q, "qdot": qdot, "qddot": qdd_sol, "qdot_cmd": qdot_cmd,
                    "tcp": {
                        "p": p, "v": v, "w": w, "rvec": rotvec,
                        "p_ref": p_ref, "v_ref": v_ref, "w_ref": w_ref,
                        "rvec_ref": RR.from_matrix(R_ref).as_rotvec(),
                        "e_p": e_p, "e_v": e_v, "e_w": e_w, "e_r": e_r,
                    },
                    "obj_break": {"imp": imp_term, "reg": reg_term, "total": obj_total},
                })

            except Exception as ex:
                print(f"[SingleArmQP] control error: {type(ex).__name__}: {ex}")
                try:
                    self.robot.speedStop()
                except Exception:
                    pass
                break

            # perf/pacing
            elapsed = time.perf_counter() - loop_t0
            win_loop_t += elapsed
            win_iters += 1
            total_iters += 1
            if elapsed > dt + 1e-4:
                win_deadline_miss += 1
                total_deadline_miss += 1

            now = time.perf_counter()
            if now - last_log >= 1.0 and win_iters > 0:
                avg_period = win_loop_t / win_iters
                avg_hz = (1.0 / avg_period) if avg_period > 0 else float('nan')
                avg_solver_ms = (win_solver_t / win_iters) * 1000.0
                miss_pct = 100.0 * win_deadline_miss / win_iters
                print(f"[SingleArmQP] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms/iter | miss {miss_pct:4.1f}%")
                win_iters = 0; win_loop_t = 0.0; win_solver_t = 0.0; win_deadline_miss = 0
                last_log = now

            time.sleep(max(0, dt - elapsed))
            i += 1
            t_idx += 1

        # graceful stop + summary
        try:
            self.robot.speedStop()
        except Exception:
            pass
        total_time = time.perf_counter() - ctrl_t0
        overall_hz = (total_iters / total_time) if total_time > 0 else float('nan')
        overall_solver_ms = (total_solver_t / max(total_iters, 1)) * 1000.0
        overall_miss_pct = 100.0 * total_deadline_miss / max(total_iters, 1)
        print(f"[SingleArmQP SUMMARY] {overall_hz:6.2f} Hz over {total_iters} iters in {total_time:.2f}s | "
              f"solver avg {overall_solver_ms:.2f} ms | miss {overall_miss_pct:.1f}%")

    def stop(self):
        self._should_stop.set()

    # ----- plotting -----
    def plot_taskspace(self, title_prefix="TaskspaceTracking_SingleArm"):
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = [d["t"] for d in self.control_data]
        t0 = ts[0]
        t = np.array(ts) - t0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X", "Y", "Z"]

        P        = np.array([d["tcp"]["p"]        for d in self.control_data])
        V        = np.array([d["tcp"]["v"]        for d in self.control_data])
        RVEC     = np.array([d["tcp"]["rvec"]     for d in self.control_data])
        W        = np.array([d["tcp"]["w"]        for d in self.control_data])
        P_ref    = np.array([d["tcp"]["p_ref"]    for d in self.control_data])
        V_ref    = np.array([d["tcp"]["v_ref"]    for d in self.control_data])
        RVEC_ref = np.array([d["tcp"]["rvec_ref"] for d in self.control_data])
        W_ref    = np.array([d["tcp"]["w_ref"]    for d in self.control_data])

        E_p = np.array([d["tcp"]["e_p"] for d in self.control_data])
        E_v = np.array([d["tcp"]["e_v"] for d in self.control_data])
        E_w = np.array([d["tcp"]["e_w"] for d in self.control_data])
        E_r = np.array([d["tcp"]["e_r"] for d in self.control_data])

        E_p_mag = np.linalg.norm(E_p, axis=1)
        E_v_mag = np.linalg.norm(E_v, axis=1)
        E_w_mag = np.linalg.norm(E_w, axis=1)
        E_r_mag = np.linalg.norm(E_r, axis=1)

        fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
        fig.suptitle(f"{title_prefix} – {timestamp_str}", fontsize=14, y=0.99)

        ax = axes[0, 0]
        for k in range(3):
            ax.plot(t, P[:, k], color=colors[k], label=f"p{labels[k]}")
            ax.plot(t, P_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"p{labels[k]} ref")
        ax.set_title("Position vs Ref"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[0, 1]
        for k in range(3):
            ax.plot(t, E_p[:, k], color=colors[k], label=f"e_p{labels[k]}")
        ax.plot(t, E_p_mag, "-", linewidth=2.0, color="black", label="‖e_p‖")
        ax.set_title("Translational Error"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[1, 0]
        for k in range(3):
            ax.plot(t, RVEC[:, k], color=colors[k], label=f"r{labels[k]}")
            ax.plot(t, RVEC_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"r{labels[k]} ref")
        ax.set_title("Rotation (rotvec) vs Ref"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[1, 1]
        for k in range(3):
            ax.plot(t, E_r[:, k], color=colors[k], label=f"e_r{labels[k]}")
        ax.plot(t, E_r_mag, "-", linewidth=2.0, color="black", label="‖e_r‖")
        ax.set_title("Rotational Error"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[2, 0]
        for k in range(3):
            ax.plot(t, V[:, k], color=colors[k], label=f"v{labels[k]}")
            ax.plot(t, V_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"v{labels[k]} ref")
        ax.set_title("Linear Velocity vs Ref"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[2, 1]
        for k in range(3):
            ax.plot(t, E_v[:, k], color=colors[k], label=f"e_v{labels[k]}")
        ax.plot(t, E_v_mag, "-", linewidth=2.0, color="black", label="‖e_v‖")
        ax.set_title("Linear-Velocity Error"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[3, 0]
        for k in range(3):
            ax.plot(t, W[:, k], color=colors[k], label=f"w{labels[k]}")
            ax.plot(t, W_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"w{labels[k]} ref")
        ax.set_title("Angular Velocity vs Ref"); ax.set_xlabel("s"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[3, 1]
        for k in range(3):
            ax.plot(t, E_w[:, k], color=colors[k], label=f"e_w{labels[k]}")
        ax.plot(t, E_w_mag, "-", linewidth=2.0, color="black", label="‖e_w‖")
        ax.set_title("Angular-Velocity Error"); ax.set_xlabel("s"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        plt.tight_layout(rect=[0, 0.035, 1, 0.97])
        fname = f"plots/{title_prefix.lower()}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_jointspace(self, title_prefix="QP_and_Jointspace_SingleArm"):
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = [d["t"] for d in self.control_data]
        t = np.array(ts) - ts[0]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        Q         = np.vstack([d["q"]         for d in self.control_data])
        QDOT      = np.vstack([d["qdot"]      for d in self.control_data])
        QDDOT     = np.vstack([d["qddot"]     for d in self.control_data])
        QDOT_cmd  = np.vstack([d["qdot_cmd"]  for d in self.control_data])
        status    = [d.get("status","") for d in self.control_data]

        n = Q.shape[1]
        jlabels = [f"J{i+1}" for i in range(n)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        fig.suptitle(f"{title_prefix} – {timestamp_str}", fontsize=14, y=0.98)

        ax = axes[0, 0]
        for j in range(n):
            ax.plot(t, Q[:, j], label=f"{jlabels[j]} meas")
        ax.set_title("Joint Positions"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(n, 6), fontsize=8)

        ax = axes[0, 1]
        for j in range(n):
            ax.plot(t, QDOT[:, j], label=f"{jlabels[j]} meas")
            ax.plot(t, QDOT_cmd[:, j], "--", alpha=0.8, label=f"{jlabels[j]} cmd")
        ax.set_title("Joint Velocities (meas vs cmd)"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(2*n, 6), fontsize=8)

        ax = axes[1, 0]
        for j in range(n):
            ax.plot(t, QDDOT[:, j], label=jlabels[j])
        ax.set_title("Joint Accelerations (QP qdd)"); ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²")
        ax.grid(True, alpha=0.3); ax.legend(ncol=min(n, 6), fontsize=8)

        def classify_status(s):
            s = (s or "").lower()
            if s == "optimal": return 0
            if s in ("optimal_inaccurate","user_limit","max_iters_reached",
                     "iteration_limit_reached","user_limit_reached"): return 1
            if any(k in s for k in ["infeasible","unbounded","solver_error","error",
                                    "dual_infeasible","primal_infeasible"]): return 2
            return 2

        class_vals = np.array([classify_status(s) for s in status], dtype=float)
        ax = axes[1, 1]
        ax.step(t, class_vals, where="post")
        ax.set_title("QP status (0=good,1=warn,2=bad)")
        ax.set_xlabel("time [s]"); ax.set_yticks([0,1,2]); ax.set_yticklabels(["good","warn","bad"])
        ax.set_ylim(-0.5, 2.5); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        mask = np.array([("obj_break" in d) for d in self.control_data], dtype=bool)
        if not np.any(mask):
            print("No obj_break logged."); return

        imp  = np.array([d["obj_break"]["imp"]     for d in self.control_data if "obj_break" in d])
        reg  = np.array([d["obj_break"]["reg"]     for d in self.control_data if "obj_break" in d])
        tot  = np.array([d["obj_break"]["total"]   for d in self.control_data if "obj_break" in d])
        t_o  = t[mask]

        status = [d.get("status","") for d in self.control_data]
        def classify_status(s):
            s = (s or "").lower()
            if s == "optimal": return 0
            if s in ("optimal_inaccurate","user_limit","max_iters_reached",
                     "iteration_limit_reached","user_limit_reached"): return 1
            if any(k in s for k in ["infeasible","unbounded","solver_error","error",
                                    "dual_infeasible","primal_infeasible"]): return 2
            return 2
        sev = np.array([classify_status(s) for s in status], dtype=float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        ax1.plot(t_o, tot, label="total", linewidth=2)
        ax1.plot(t_o, imp, label="impedance")
        ax1.plot(t_o, reg, label="regularizer")
        ax1.set_ylabel("objective"); ax1.grid(True, alpha=0.3); ax1.legend(loc="best", fontsize=8)
        ax1.set_title("QP objective breakdown", fontsize=11)

        ax2.step(t, sev, where="post", linewidth=2, color="black")
        ax2.axhspan(-0.5, 0.5, facecolor="green",  alpha=0.08)
        ax2.axhspan(0.5, 1.5, facecolor="yellow", alpha=0.10)
        ax2.axhspan(1.5, 2.5, facecolor="red",    alpha=0.08)
        ax2.set_yticks([0,1,2]); ax2.set_yticklabels(["GOOD","WARN","BAD"])
        ax2.set_ylim(-0.5, 2.5); ax2.set_xlabel("time [s]")
        ax2.set_title("QP solve health", fontsize=11)
        ax2.grid(True, axis="x", alpha=0.3)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fname}")


# ===========================
# example main
# ===========================

if __name__ == "__main__":
    # impedance gains (tune rotation lower)
    K = np.diag([400, 400, 400, 20, 20, 20])
    robot = URImpedanceController("192.168.1.33", K=K)

    # diagonal task weighting
    W_imp = diag6([5e4, 5e4, 5e4, 0, 0, 0])

    traj_path = "motion_planner/trajectories/pick_and_place.npz"
    Hz = 75

    ctrl = SingleArmImpedanceQP(
        robot=robot,
        Hz=Hz,
        trajectory_npz_path=traj_path,
        W_imp=W_imp,
        lambda_reg=5e-15,        # regularization on qdd
        tikhonov_lambda=1e-1,   # Λ regularization
    )

    try:
        # robot.wait_for_commands()
        # robot.go_home()
        # robot.wait_for_commands()
        # robot.wait_until_done()

        robot.go_to_approach()
        robot.wait_for_commands()
        robot.wait_until_done()

        ctrl.run()

        ctrl.plot_taskspace()
        ctrl.plot_jointspace()
        ctrl.plot_qp_objective()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        try:
            ctrl.stop()
            robot.stop_control()
        except Exception:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            ctrl.stop()
            robot.stop_control()
        except Exception:
            pass
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Robot disconnected.")
