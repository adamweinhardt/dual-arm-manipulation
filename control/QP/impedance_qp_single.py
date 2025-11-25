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
import matplotlib.pyplot as plt
from numpy.linalg import solve, eigh
# project deps
from control.PID.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity, _as_rowvec_1d, is_finite, short_arc_log, diag6


class URImpedanceController(URForceController):
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

    def get_J_ur(self, q):
        J_flat = self.rtde_control.getJacobian(q.tolist())
        J_base = np.array(J_flat, dtype=float).reshape((6, 6))
        return J_base

    def get_Jdot_ur(self, q, v):
        Jdot_flat = self.rtde_control.getJacobianTimeDerivative(q.tolist(), v.tolist())
        Jdot_base = np.array(Jdot_flat, dtype=float).reshape((6, 6))
        return Jdot_base

    def get_M_ur(self, q):
        M_flat = self.rtde_control.getMassMatrix(q.tolist(), include_rotors_inertia=True)
        M = np.array(M_flat, dtype=float).reshape((6, 6))
        return 0.5 * (M + M.T)
    
    def get_J(self, q):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(self.pin_model, self.pin_data, self.pin_frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return self.get_J_pin(J)

    def get_Jdot(self, q, v):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, self.pin_frame_id,
                                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return self.get_J_pin(dJ)
    
    def get_M(self, q):
        M = pin.crba(self.pin_model, self.pin_data, q)
        return np.asarray(0.5 * (M + M.T))

    def get_Lambda_and_inv(self, J, M,
                        rel_reg: float = 1e-3,
                        abs_reg: float = 1e-3,
                        return_info: bool = False):
        """
        Robust computation of operational-space inertia and its inverse.

            Λ      = (J M^{-1} J^T)^-1
            Λ^{-1} =  J M^{-1} J^T   (with eigenvalue-based regularization)

        Steps:
        1) X = M^{-1} J^T              via solve(M, J.T)
        2) A = J X = J M^{-1} J^T
        3) symmetrize A
        4) eigendecompose A
        5) clamp small eigenvalues
        6) reconstruct:
            Λ^{-1} = Q diag(eigvals_reg) Q^T
            Λ      = Q diag(1/eigvals_reg) Q^T
        """

        # 1) Solve M X = J^T instead of forming M^{-1}
        X = solve(M, J.T)          # shape: (nq, 6)

        # 2) A = J M^{-1} J^T = J X
        A = J @ X                  # shape: (6, 6)

        # 3) Symmetrize numerically
        A = 0.5 * (A + A.T)

        # 4) Eigen-decompose A (symmetric → eigh)
        eigvals, Q = eigh(A)       # eigvals real, Q orthonormal

        lam_max = float(np.max(eigvals))
        # relative + absolute floor
        floor = max(abs_reg, rel_reg * lam_max)

        # 5) Regularize eigenvalues
        eigvals_reg = np.maximum(eigvals, floor)

        # 6a) Λ^{-1} = A_reg = Q diag(eigvals_reg) Q^T
        Lambda_inv = (Q * eigvals_reg) @ Q.T
        Lambda_inv = 0.5 * (Lambda_inv + Lambda_inv.T)

        # 6b) Λ = A_reg^{-1} = Q diag(1/eigvals_reg) Q^T
        inv_eigs = 1.0 / eigvals_reg
        Lambda = (Q * inv_eigs) @ Q.T
        Lambda = 0.5 * (Lambda + Lambda.T)

        if return_info:
            info = {
                "eigvals_A": eigvals,
                "eigvals_A_reg": eigvals_reg,
                "lam_max": lam_max,
                "lam_min": float(np.min(eigvals)),
                "lam_min_reg": float(np.min(eigvals_reg)),
                "cond_A": lam_max / max(float(np.min(np.abs(eigvals))), 1e-12),
            }
            return Lambda, Lambda_inv, info

        return Lambda, Lambda_inv
    
    def get_D(self, K, Lambda):
        K = 0.5 * (K + K.T)
        L = 0.5 * (Lambda + Lambda.T)

        # --- sqrt(K) safely ---
        wK, VK = np.linalg.eigh(K)
        wK = np.maximum(wK, 1e-12)                  # tiny floor
        S_K = (VK * np.sqrt(wK)) @ VK.T
        S_K = 0.5 * (S_K + S_K.T)

        # --- sqrt(Lambda) safely ---
        wL, VL = np.linalg.eigh(L)
        wL = np.maximum(wL, 1e-12)
        S_L = (VL * np.sqrt(wL)) @ VL.T
        S_L = 0.5 * (S_L + S_L.T)

        # --- D = S_L S_K + S_K S_L ---
        D = S_L @ S_K + S_K @ S_L
        D = 0.5 * (D + D.T)

        return D


    def wrench_desired(self, K, D, e_p, e_r, e_v, e_w):
        X_err = np.hstack((e_p, e_r))
        Xd_err = np.hstack((e_v, e_w))
        return D @ Xd_err + K @ X_err

class SingleArmImpedanceQP:
    def __init__(
        self,
        robot: URImpedanceController,
        Hz: int,
        trajectory_npz_path: str,
        W_imp=None,
        lambda_reg: float = 1e-6,
        tikhonov_lambda: float = 1e-6,
    ):
        self.robot = robot
        self.Hz = int(Hz)
        self.dt = 1.0 / float(self.Hz)

        # refs
        self._npz_path = trajectory_npz_path

        # symmetric limits
        self.joint_pose_limit  = np.deg2rad(360.0)
        self.joint_speed_limit = np.deg2rad(180.0)
        self.joint_accel_limit = np.deg2rad(120.0)

        # weights
        self.W_imp_np = np.asarray(W_imp, dtype=float)
        self.lambda_reg = float(lambda_reg)
        self.tikhonov_lambda = float(tikhonov_lambda)

        # buffers
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

        state = self.robot.get_state()
        p0 = np.array(state["pose"][:3])
        R0 = RR.from_rotvec(state["pose"][3:6]).as_matrix()

        self.p_ref = np.zeros_like(p_box)
        self.v_ref = np.zeros_like(v_box)
        self.R_ref = np.zeros_like(R_box)
        self.w_ref = np.zeros_like(w_box)

        R_box0 = R_box[0]
        for t in range(self.traj_len):
            self.p_ref[t] = p0 + p_box[t]
            self.v_ref[t] = v_box[t]
            R_rel = R_box[t] @ R_box0.T
            self.R_ref[t] = R_rel @ R0
            self.w_ref[t] = w_box[t]

    # ----- QP model -----
    def _build_qp(self, dt):
        n = 6
        self.qdd = cp.Variable(n, name="qdd")

        self.J_p     = cp.Parameter((6, n), name="J")
        self.Jdot_p  = cp.Parameter((6, n), name="Jdot")
        self.q_p     = cp.Parameter(n,      name="q")
        self.qdot_p  = cp.Parameter(n,      name="qdot")
        self.a_des_p = cp.Parameter(6,      name="a_des")
        self.W_imp_p = cp.Parameter((6, 6), name="W_imp")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(0.5 * dt * dt))

        q_next    = self.q_p    + self.qdot_p * self.dt_c + self.qdd * self.dt2_c
        qdot_next = self.qdot_p + self.qdd * self.dt_c

        e_imp = self.J_p @ self.qdd + self.Jdot_p @ self.qdot_p - self.a_des_p
        obj = cp.sum_squares(self.W_imp_p @ e_imp) + self.lambda_reg * cp.sum_squares(self.qdd)

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
            eps_abs=1e-6,
            eps_rel=1e-6,
            alpha=1.6,
            max_iter=10000,
            adaptive_rho=True,
            adaptive_rho_interval=45,
            polish=True,
            check_termination=10,
            warm_start=True,
        )

    def log_qp_tick(self, *,
                    solve_dt,
                    q, qdot, qdd_sol, dt,
                    J, Jdot, a_des,
                    t_model=None, t_J=None, t_Jdot=None, t_M=None, t_Lam=None, t_D=None, t_loop=None,
                    p=None, v=None, w=None, R=None,
                    p_ref=None, v_ref=None, w_ref=None, R_ref=None,
                    M=None):
        import types

        def _get(obj, key, default=np.nan):
            try:
                if obj is None: return default
                if isinstance(obj, dict): return obj.get(key, default)
                if isinstance(obj, types.SimpleNamespace): return getattr(obj, key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        status_str = (getattr(self.prob, "status", "") or "").lower()
        stats      = getattr(self.prob, "solver_stats", None)
        extra      = _get(stats, "extra_stats", None)

        iters      = _get(stats, "num_iters",    np.nan)
        solve_time = _get(stats, "solve_time",   solve_dt if np.isfinite(solve_dt) else np.nan)
        setup_time = _get(stats, "setup_time",   np.nan)
        max_iter   = float(self.prob_kwargs.get("max_iter", np.nan))

        pri_res = _get(extra, "primal_residual", np.nan)
        if not np.isfinite(pri_res): pri_res = _get(extra, "pri_res", np.nan)
        dua_res = _get(extra, "dual_residual",   np.nan)
        if not np.isfinite(dua_res): dua_res = _get(extra, "dua_res", np.nan)

        obj_val = _get(extra, "objective", np.nan)
        if not np.isfinite(obj_val):
            try:
                obj_val = float(self.prob.value) if self.prob.value is not None else np.nan
            except Exception:
                obj_val = np.nan

        rho_updates = _get(extra, "rho_updates", np.nan)
        if not np.isfinite(rho_updates): rho_updates = _get(extra, "n_rho_update", np.nan)
        polish = _get(extra, "polish", None)
        if polish in (None, np.nan): polish = _get(extra, "polish_status", None)

        solver_fb = {
            "status": status_str,
            "iters": iters,
            "solve_time": solve_time,
            "setup_time": setup_time,
            "pri_res": pri_res,
            "dua_res": dua_res,
            "obj": obj_val,
            "rho_updates": rho_updates,
            "polish": polish,
            "max_iter": max_iter,
        }

        a_pred = J @ qdd_sol + Jdot @ qdot
        e_imp  = J @ qdd_sol + Jdot @ qdot - a_des
        imp    = float((self.W_imp_np @ e_imp).T @ (self.W_imp_np @ e_imp))
        reg    = float(self.lambda_reg) * float(qdd_sol @ qdd_sol)

        # saturations
        try:
            q_next    = q + qdot * dt + 0.5 * qdd_sol * dt * dt
            qdot_next = qdot + qdd_sol * dt
            tol = 1e-6
            pos_sat = float(np.mean(np.abs(q_next)    >= (self.joint_pose_limit  - tol)))
            vel_sat = float(np.mean(np.abs(qdot_next) >= (self.joint_speed_limit - tol)))
            acc_sat = float(np.mean(np.abs(qdd_sol)   >= (self.joint_accel_limit - tol)))
        except Exception:
            pos_sat = vel_sat = acc_sat = np.nan

        # conditioning (optional)
        cond_A = np.nan; lam_min_A = np.nan
        if M is not None and np.all(np.isfinite(M)):
            try:
                Minv = np.linalg.inv(M)
                A = J @ Minv @ J.T
                A = 0.5 * (A + A.T)
                ew = np.linalg.eigvalsh(A)
                lam_min_A = float(np.clip(np.min(ew), 0.0, np.inf))
                lam_max_A = float(np.max(ew)) if ew.size else np.nan
                cond_A = float(lam_max_A / max(lam_min_A, 1e-12)) if np.isfinite(lam_max_A) else np.nan
            except Exception:
                cond_A = lam_min_A = np.nan

        tcp = None
        if (p is not None) and (R is not None) and (p_ref is not None) and (R_ref is not None):
            try:
                e_p = p_ref - p
                e_v = (v_ref - v) if (v is not None and v_ref is not None) else np.full(3, np.nan)
                e_w = (w_ref - w) if (w is not None and w_ref is not None) else np.full(3, np.nan)
                q_ref = RR.from_matrix(R_ref).as_quat()
                q_cur = RR.from_matrix(R).as_quat()
                if np.dot(q_ref, q_cur) < 0.0: q_ref = -q_ref
                e_r = short_arc_log(R_ref, R)
                tcp = {
                    "p": p, "v": v, "w": w, "rvec": RR.from_matrix(R).as_rotvec(),
                    "p_ref": p_ref, "v_ref": v_ref, "w_ref": w_ref, "rvec_ref": RR.from_matrix(R_ref).as_rotvec(),
                    "e_p": e_p, "e_v": e_v, "e_w": e_w, "e_r": e_r,
                }
            except Exception:
                tcp = None

        self.control_data.append({
            "t": time.time(),
            "status": status_str,
            "solver_fb": solver_fb,
            "q": q, "qdot": qdot, "qddot": qdd_sol,
            "qdot_cmd": (qdot + qdd_sol * dt),
            "obj_break": {"imp": imp, "reg": reg, "total": imp + reg},
            "dynamics": {"a_des": a_des, "a_pred": a_pred,
                         "model_time": t_model if t_model is not None else np.nan,
                         "solver_time": solve_dt if solve_dt is not None else np.nan},
            "perf": {"loop_s": t_loop if t_loop is not None else np.nan,
                     "model_s": t_model if t_model is not None else np.nan,
                     "solver_s": solve_dt if solve_dt is not None else np.nan,
                     "t_J": t_J if t_J is not None else np.nan,
                     "t_Jdot": t_Jdot if t_Jdot is not None else np.nan,
                     "t_M": t_M if t_M is not None else np.nan,
                     "t_Lam": t_Lam if t_Lam is not None else np.nan,
                     "t_D": t_D if t_D is not None else np.nan},
            "diag": {"cond_A": cond_A, "lam_min_A": lam_min_A,
                     "pos_sat": pos_sat, "vel_sat": vel_sat, "acc_sat": acc_sat},
            "tcp": tcp
        })

    # ----- control loop -----
    def run(self):
        self._load_refs()
        time.sleep(0.1)

        dt = self.dt
        t_idx = 0
        self._should_stop.clear()

        print("[SingleArmQP] Starting impedance QP control loop")

        ctrl_t0 = time.perf_counter()
        i = 0

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
                p = np.array(state["pose"][:3])
                v = np.array(state["speed"][:3])
                rotvec = np.array(state["pose"][3:6])
                R = RR.from_rotvec(rotvec).as_matrix()
                w = np.array(state["speed"][3:6])
                q    = self.robot.get_q()
                qdot = self.robot.get_qdot()

                # model timings
                t_model_start = time.perf_counter()
                t0 = time.perf_counter(); J    = self.robot.get_J(q);                 t_J = time.perf_counter() - t0
                t0 = time.perf_counter(); Jdot = self.robot.get_Jdot(q, qdot);        t_Jdot = time.perf_counter() - t0
                t0 = time.perf_counter(); M    = self.robot.get_M(q);                 t_M = time.perf_counter() - t0
                t0 = time.perf_counter(); Lam, Lam_inv, lam_info  = self.robot.get_Lambda_and_inv(J, M, return_info=True); t_Lam = time.perf_counter() - t0
                t0 = time.perf_counter(); D    = self.robot.get_D(self.robot.K, Lam); t_D = time.perf_counter() - t0
                model_dt = time.perf_counter() - t_model_start

                # errors
                e_p = p_ref - p
                e_v = v_ref - v
                e_r = short_arc_log(R_ref, R)
                e_w = w_ref - w

                # desired wrench/accel
                f_des = self.robot.wrench_desired(self.robot.K, D, e_p, e_r, e_v, e_w)
                a_des = Lam_inv @ f_des

                if not is_finite(J, Jdot, q, qdot, a_des):
                    raise ValueError("Non-finite detected in inputs to QP")

                # QP params
                self.J_p.value = _freeze_sparsity(J)
                self.Jdot_p.value = _freeze_sparsity(Jdot)
                self.q_p.value = q
                self.qdot_p.value = qdot
                self.a_des_p.value = a_des
                self.W_imp_p.value = self.W_imp_np

                # solve
                t_solve_start = time.perf_counter()
                self.prob.solve(solver=cp.OSQP, **self.prob_kwargs)
                solve_dt = time.perf_counter() - t_solve_start

                status = (self.prob.status or "").lower()
                status_ok = status in ("optimal", "optimal_inaccurate")
                qdd_sol = _as_rowvec_1d(self.qdd.value, "qdd", length=6) if status_ok and self.qdd.value is not None else np.zeros(6)

                # command
                qdot_cmd = qdot + qdd_sol * dt
                self.robot.speedJ(qdot_cmd.tolist(), dt)

                                # ------------------------------------------
                # DEEP DEBUG BLOCK (NO PINOCCHIO)
                # ------------------------------------------
                if (i % 75) == 0:
                    print("\n" + "="*60)
                    print(f"[DBG] iter {i}, t = {i*dt:.3f} s")
                    print("="*60)

                    # 0) Basic error norms
                    print("[DBG] Errors:")
                    print(f"  ‖e_p‖ = {np.linalg.norm(e_p):.6f} m")
                    print(f"  ‖e_v‖ = {np.linalg.norm(e_v):.6f} m/s")
                    print(f"  ‖e_r‖ = {np.linalg.norm(e_r):.6f} rad")
                    print(f"  ‖e_w‖ = {np.linalg.norm(e_w):.6f} rad/s")

                    # 1) State vs reference (components)
                    print("[DBG] Position (world):")
                    print(f"  p     = {p}")
                    print(f"  p_ref = {p_ref}")
                    print(f"  e_p   = {e_p}")

                    print("[DBG] Linear velocity (world):")
                    print(f"  v     = {v}")
                    print(f"  v_ref = {v_ref}")
                    print(f"  e_v   = {e_v}")

                    print("[DBG] Angular velocity (world):")
                    print(f"  w     = {w}")
                    print(f"  w_ref = {w_ref}")
                    print(f"  e_w   = {e_w}")

                    # 2) Jacobian sanity
                    print("[DBG] Jacobian checks (UR):")
                    xdot_pred = J @ qdot
                    v_pred, w_pred = xdot_pred[:3], xdot_pred[3:]
                    v_meas, w_meas = v, w

                    print(f"  J shape   = {J.shape}, Jdot shape = {Jdot.shape}")
                    print(f"  J @ qdot (lin) = {v_pred}")
                    print(f"  measured v     = {v_meas}")
                    print(f"  lin diff       = {v_pred - v_meas}")
                    print(f"  ‖lin diff‖     = {np.linalg.norm(v_pred - v_meas):.6e}")

                    print(f"  J @ qdot (ang) = {w_pred}")
                    print(f"  measured w     = {w_meas}")
                    print(f"  ang diff       = {w_pred - w_meas}")
                    print(f"  ‖ang diff‖     = {np.linalg.norm(w_pred - w_meas):.6e}")

                    lin_ratio = np.linalg.norm(v_pred) / (np.linalg.norm(v_meas) + 1e-12)
                    ang_ratio = np.linalg.norm(w_pred) / (np.linalg.norm(w_meas) + 1e-12)
                    print(f"  ‖v_pred‖ / ‖v_meas‖ = {lin_ratio:.3f}")
                    print(f"  ‖w_pred‖ / ‖w_meas‖ = {ang_ratio:.3f}")
                    print("  (Expect these ratios ≈ 1 if frames/units are consistent.)")

                    # 3) Mass matrix + Lambda checks (UR only)
                    print("[DBG] Mass matrix M (UR):")
                    print(f"  M =\n{M}")
                    sym_M = 0.5 * (M + M.T)
                    print(f"  ‖M - M.T‖_F = {np.linalg.norm(M - M.T):.6e}")
                    evals_M = np.linalg.eigvalsh(sym_M)
                    print(f"  eig(M)      = {evals_M}")
                    print(f"  min eig(M)  = {np.min(evals_M):.3e}")
                    print(f"  max eig(M)  = {np.max(evals_M):.3e}")

                    print("[DBG] Lambda (UR):")
                    sym_Lam = 0.5 * (Lam + Lam.T)
                    eig_L = np.linalg.eigvalsh(sym_Lam)
                    print(f"  Λ =\n{Lam}")
                    print(f"  eig(Λ)      = {eig_L}")
                    print(f"  min eig(Λ)  = {np.min(eig_L):.3e}")
                    print(f"  max eig(Λ)  = {np.max(eig_L):.3e}")
                    condL = np.max(eig_L) / max(np.min(eig_L), 1e-12)
                    print(f"  cond(Λ)     = {condL:.3e}")

                    eA      = lam_info["eigvals_A"]
                    eA_reg  = lam_info["eigvals_A_reg"]

                    min_eΛ = 1.0 / np.max(eA_reg)
                    max_eΛ = 1.0 / np.min(eA_reg)

                    print("[DBG] Lambda (UR):")
                    print("  eig(A)      =", eA)
                    print("  eig(A_reg)  =", eA_reg)
                    print("  min eig(A)  =", lam_info["lam_min"])
                    print("  max eig(A)  =", lam_info["lam_max"])
                    print("  cond(A)     =", lam_info["cond_A"])
                    print("  min eig(Λ)  =", min_eΛ)
                    print("  max eig(Λ)  =", max_eΛ)


                    # 4) Impedance / wrench checks
                    X_err  = np.hstack((e_p, e_r))
                    Xd_err = np.hstack((e_v, e_w))
                    f_des_re = D @ Xd_err + self.robot.K @ X_err

                    print("[DBG] Wrench / impedance:")
                    print(f"  K diag        = {np.diag(self.robot.K)}")
                    print(f"  D (6x6)       =\n{D}")
                    print(f"  f_des         = {f_des}")
                    print(f"  f_des_recomp  = {f_des_re}")
                    print(f"  ‖f_des - f_des_re‖ = {np.linalg.norm(f_des - f_des_re):.6e}")
                    print("  (This should be ≈ 0 — otherwise bug in wrench_desired / error stacking.)")

                    # 5) QP / acceleration checks
                    a_pred = J @ qdd_sol + Jdot @ qdot
                    e_imp  = a_pred - a_des

                    print("[DBG] Acceleration checks:")
                    print(f"  a_des  (Λ⁻¹·wrench) = {a_des}")
                    print(f"  a_pred (J q̈* + J̇ q̇) = {a_pred}")
                    print(f"  e_imp  = a_pred - a_des = {e_imp}")
                    print(f"  ‖e_imp‖ = {np.linalg.norm(e_imp):.6e}")

                    We = self.W_imp_np @ e_imp
                    imp_term = float(We.T @ We)
                    reg_term = float(self.lambda_reg) * float(qdd_sol @ qdd_sol)
                    print(f"  obj_imp (recomputed)   = {imp_term:.6e}")
                    print(f"  obj_reg (recomputed)   = {reg_term:.6e}")
                    print(f"  obj_total (recomputed) = {imp_term + reg_term:.6e}")

                    # 6) Joint limits & saturation
                    q_next    = q + qdot * dt + 0.5 * qdd_sol * dt * dt
                    qdot_next = qdot + qdd_sol * dt

                    print("[DBG] Joint states:")
                    print(f"  q         = {q}")
                    print(f"  qdot      = {qdot}")
                    print(f"  qdd_sol   = {qdd_sol}")
                    print(f"  q_next    = {q_next}")
                    print(f"  qdot_next = {qdot_next}")

                    print("[DBG] Joint limit usage (fraction of limit):")
                    print(f"  max |q|      / q_pos_lim  = {np.max(np.abs(q)      / self.joint_pose_limit):.3f}")
                    print(f"  max |q_next| / q_pos_lim  = {np.max(np.abs(q_next) / self.joint_pose_limit):.3f}")
                    print(f"  max |qdot|   / q_vel_lim  = {np.max(np.abs(qdot)   / self.joint_speed_limit):.3f}")
                    print(f"  max |qdd|    / q_acc_lim  = {np.max(np.abs(qdd_sol)/ self.joint_accel_limit):.3f}")

                    # 7) Solver feedback
                    status = (self.prob.status or "").lower()
                    print("[DBG] QP solver:")
                    print(f"  status   = {status}")
                    print(f"  solve_dt = {solve_dt*1000.0:.3f} ms")
                    print(f"  model_dt = {model_dt*1000.0:.3f} ms")
                    print(f"  t_J   = {t_J*1000.0:.3f} ms")
                    print(f"  t_Jdot= {t_Jdot*1000.0:.3f} ms")
                    print(f"  t_M   = {t_M*1000.0:.3f} ms")
                    print(f"  t_Lam = {t_Lam*1000.0:.3f} ms")
                    print(f"  t_D   = {t_D*1000.0:.3f} ms")
                    print("="*60)


                # log
                self.log_qp_tick(
                    solve_dt=solve_dt,
                    q=q, qdot=qdot, qdd_sol=qdd_sol, dt=dt,
                    J=J, Jdot=Jdot, a_des=a_des,
                    t_model=model_dt, t_J=t_J, t_Jdot=t_Jdot, t_M=t_M, t_Lam=t_Lam, t_D=t_D,
                    t_loop=(time.perf_counter() - loop_t0),
                    p=p, v=v, w=w, R=R,
                    p_ref=p_ref, v_ref=v_ref, w_ref=w_ref, R_ref=R_ref,
                    M=M
                )

            except Exception as ex:
                print(f"[SingleArmQP] control error: {type(ex).__name__}: {ex}")
                try:
                    self.robot.speedStop()
                except Exception:
                    pass
                break

            # pacing
            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0, dt - elapsed))
            i += 1
            t_idx += 1

        try:
            self.robot.speedStop()
        except Exception:
            pass
        total_time = time.perf_counter() - ctrl_t0
        actual_hz = i / total_time if total_time > 0 else float("nan")
        print(f"[SingleArmQP SUMMARY] iters={i} | time={total_time:.2f}s | ~{actual_hz:.2f} Hz")


    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        """Objective-only: total vs impedance vs regularizer."""
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        mask = np.array([("obj_break" in d) for d in self.control_data], dtype=bool)
        if not np.any(mask):
            print("No obj_break logged."); return

        imp  = np.array([d["obj_break"]["imp"]   for d in self.control_data if "obj_break" in d])
        reg  = np.array([d["obj_break"]["reg"]   for d in self.control_data if "obj_break" in d])
        tot  = np.array([d["obj_break"]["total"] for d in self.control_data if "obj_break" in d])
        t_o  = t[mask]

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)
        ax.plot(t_o, tot, label="total", linewidth=2)
        ax.plot(t_o, imp, label="impedance")
        ax.plot(t_o, reg, label="regularizer")
        ax.set_ylabel("objective"); ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
        ax.set_title("QP objective breakdown", fontsize=11)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_qp_performance(self, title_prefix="QP_Performance"):
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)
        import matplotlib.gridspec as gridspec

        ts = np.array([d["t"] for d in self.control_data], dtype=float)
        t = ts - ts[0]

        def _get(d, dyn_key, perf_key):
            if "dynamics" in d and dyn_key in d["dynamics"]:
                return d["dynamics"][dyn_key]
            if "perf" in d and perf_key in d["perf"]:
                return d["perf"][perf_key]
            return np.nan

        model_s  = np.array([_get(d, "model_time",  "model_s")  for d in self.control_data], dtype=float)
        solver_s = np.array([_get(d, "solver_time", "solver_s") for d in self.control_data], dtype=float)
        loop_s   = np.array([_get(d, None,          "loop_s")   for d in self.control_data], dtype=float)

        model_ms, solver_ms, loop_ms = model_s*1000.0, solver_s*1000.0, loop_s*1000.0

        finite_any = np.isfinite(model_s) | np.isfinite(solver_s) | np.isfinite(loop_s)
        tmin, tmax = (t[np.where(finite_any)[0][0]], t[np.where(finite_any)[0][-1]]) if finite_any.any() else (t[0], t[-1])

        def rolling_mean(a, w):
            if len(a) < 2: return a
            w = max(3, min(int(w), len(a)))
            a2 = np.array(a, dtype=float)
            med = np.nanmedian(a2)
            a2[~np.isfinite(a2)] = med if np.isfinite(med) else 0.0
            return np.convolve(a2, np.ones(w)/w, mode="same")

        loop_hz   = 1.0 / np.maximum(loop_s,  1e-9)
        model_hz  = 1.0 / np.maximum(model_s, 1e-9)
        solver_hz = 1.0 / np.maximum(solver_s,1e-9)
        win = max(5, int(0.25 * len(t)))
        loop_hz_rm, model_hz_rm, solver_hz_rm = [rolling_mean(x, win) for x in (loop_hz, model_hz, solver_hz)]

        TJ    = np.array([d.get("perf", {}).get("t_J",    np.nan) for d in self.control_data]) * 1000.0
        TJdot = np.array([d.get("perf", {}).get("t_Jdot", np.nan) for d in self.control_data]) * 1000.0
        TM    = np.array([d.get("perf", {}).get("t_M",    np.nan) for d in self.control_data]) * 1000.0
        TLam  = np.array([d.get("perf", {}).get("t_Lam",  np.nan) for d in self.control_data]) * 1000.0
        TD    = np.array([d.get("perf", {}).get("t_D",    np.nan) for d in self.control_data]) * 1000.0
        have_breakdown = np.isfinite([TJ, TJdot, TM, TLam, TD]).any()

        # status timeline (good/warn/bad)
        def classify_status(s):
            s = (s or "").lower()
            if s == "optimal": return 0
            if s in ("optimal_inaccurate","solved inaccurate","user_limit",
                     "max_iters_reached","iteration_limit_reached","user_limit_reached"): return 1
            if any(k in s for k in ("infeasible","unbounded","solver_error","error",
                                    "dual_infeasible","primal_infeasible")): return 2
            return np.nan

        sev = np.array([classify_status(d.get("status","")) for d in self.control_data], dtype=float)
        sev_valid = sev[np.isfinite(sev)]
        warn_pct = 100.0*np.mean(sev_valid==1) if sev_valid.size else np.nan
        bad_pct  = 100.0*np.mean(sev_valid==2) if sev_valid.size else np.nan

        # solver status counts (raw)
        statuses = [d.get("status","").lower() for d in self.control_data]
        s_counts = {s: statuses.count(s) for s in sorted(set(statuses)) if s}

        rows = 4 if have_breakdown else 3
        fig = plt.figure(figsize=(14, 4.3*rows))
        gs  = gridspec.GridSpec(rows, 1, height_ratios=[2.2, 1.8, 1.4] + ([1.2] if rows==4 else []))

        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
        ax4 = fig.add_subplot(gs[3,0]) if rows == 4 else None
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        # (1) rolling Hz
        ax1.plot(t, loop_hz_rm,   label="Loop Hz",   linewidth=2)
        ax1.plot(t, model_hz_rm,  label="Model Hz",  linewidth=2)
        ax1.plot(t, solver_hz_rm, label="Solver Hz", linewidth=2)
        if hasattr(self, "Hz") and np.isfinite(self.Hz):
            ax1.axhline(self.Hz, ls="--", color="k", alpha=0.35, label=f"Target {self.Hz:.0f} Hz")
        ax1.set_ylabel("Hz"); ax1.grid(True, alpha=0.3); ax1.legend(loc="best", fontsize=8)
        ax1.set_title("Rolling frequency"); ax1.set_xlim(tmin, tmax)

        # (2) per-iteration ms
        ax2.plot(t, model_ms,  label="Model ms",  linewidth=1.8)
        ax2.plot(t, solver_ms, label="Solver ms", linewidth=1.8)
        ax2.plot(t, loop_ms,   label="Loop period ms", linewidth=1.5, alpha=0.7)
        if hasattr(self, "dt") and np.isfinite(self.dt):
            ax2.axhline(self.dt*1000.0, ls="--", color="k", alpha=0.25, label="Target dt")
        ax2.set_ylabel("ms"); ax2.grid(True, alpha=0.3); ax2.legend(loc="best", fontsize=8)
        ax2.set_title("Per-iteration timings"); ax2.set_xlim(tmin, tmax)

        # (3) status timeline
        ax3.step(t, np.nan_to_num(sev, nan=1.5), where="post", linewidth=2, color="black")
        ax3.axhspan(-0.5, 0.5, facecolor="green",  alpha=0.1)
        ax3.axhspan(0.5, 1.5, facecolor="yellow", alpha=0.1)
        ax3.axhspan(1.5, 2.5, facecolor="red",    alpha=0.1)
        ax3.set_yticks([0,1,2]); ax3.set_yticklabels(["GOOD","WARN","BAD"])
        ax3.set_ylim(-0.5, 2.5); ax3.set_xlabel("time [s]")
        ax3.set_title("QP solve health"); ax3.grid(True, axis="x", alpha=0.3)
        ax3.set_xlim(tmin, tmax)

        # (4) avg model breakdown (non-time)
        if ax4 is not None:
            parts  = [np.nanmean(TJ), np.nanmean(TJdot), np.nanmean(TM), np.nanmean(TLam), np.nanmean(TD)]
            labels = ["J","Jdot","M","Λ","D"]
            parts  = [0.0 if not np.isfinite(v) else float(v) for v in parts]
            cum = 0.0
            for val, lab in zip(parts, labels):
                ax4.barh(["Model avg breakdown"], [val], left=cum, label=f"{lab} {val:.2f} ms")
                cum += val
            ax4.set_xlabel("ms"); ax4.grid(True, axis="x", alpha=0.3)
            ax4.set_title("Average model-time breakdown"); ax4.legend(ncol=5, fontsize=8)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # concise solver status summary (nothing else)
        print("\n[QP SUMMARY]")
        print(f"  Solver statuses: {s_counts if s_counts else '{}'}")
        print(f"  WARN%: {warn_pct:.2f}% | BAD%: {bad_pct:.2f}%")

    def print_qp_summary(self):
        """Raw solver statuses only."""
        if not self.control_data:
            print("No QP data logged."); return
        statuses = [d.get("status","").lower() for d in self.control_data]
        counts = {s: statuses.count(s) for s in sorted(set(statuses)) if s}
        print("\n[QP SUMMARY]")
        print(f"  Solver statuses: {counts if counts else '{}'}")

    def plot_jointspace(self, title_prefix="Joint_Space"):
        """Plot joint position, velocity, acceleration, and command tracking."""
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        q     = np.array([d["q"] for d in self.control_data])
        qdot  = np.array([d["qdot"] for d in self.control_data])
        qddot = np.array([d["qddot"] for d in self.control_data])
        qdot_cmd = np.array([d["qdot_cmd"] for d in self.control_data])

        n = q.shape[1] if q.ndim == 2 else 6
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(title_prefix, fontsize=13, y=0.98)

        for j in range(n):
            axs[0].plot(t, q[:, j], label=f"q{j+1}")
            axs[1].plot(t, qdot[:, j], label=f"q̇{j+1}")
            axs[2].plot(t, qddot[:, j], label=f"q̈{j+1}")
        axs[1].plot(t, qdot_cmd, '--', linewidth=1.5, alpha=0.7, label="q̇ cmd")

        axs[0].set_ylabel("Position [rad]")
        axs[1].set_ylabel("Velocity [rad/s]")
        axs[2].set_ylabel("Acceleration [rad/s²]")
        axs[2].set_xlabel("Time [s]")
        for ax in axs:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=3)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

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

    def plot_wrench_lambda_inv_minus_xddot(self, title_prefix="WrenchLambdaInvMinusXddot"):
        """
        Plot (Λ⁻¹ * wrench_desired) - (ẍ_pred) over time.
        Here ẍ_pred = J q̈* + Jdot q̇, and a_des = Λ⁻¹ * wrench_desired.
        This is the same as a_des - a_pred, using values logged in control_data['dynamics'].
        """
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        # time base
        ts = [d["t"] for d in self.control_data]
        t = np.array(ts) - ts[0]

        # pull dynamics
        try:
            A_des  = np.array([d["dynamics"]["a_des"]  for d in self.control_data])
            A_pred = np.array([d["dynamics"]["a_pred"] for d in self.control_data])
        except KeyError:
            print("Missing dynamics['a_des'] or dynamics['a_pred'] in control_data; cannot plot.")
            return

        # error = (Λ⁻¹ F_des) - ẍ_pred
        E = A_des - A_pred
        E_v = E[:, :3]   # linear accel error
        E_w = E[:, 3:]   # angular accel error
        E_v_mag = np.linalg.norm(E_v, axis=1)
        E_w_mag = np.linalg.norm(E_w, axis=1)

        # nice colors/labels
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X", "Y", "Z"]

        # figure
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
        fig.suptitle(f"{title_prefix} – {ts_str}", fontsize=14, y=0.99)

        # (1) Linear: a_des vs a_pred
        ax = axes[0, 0]
        for k in range(3):
            ax.plot(t, A_des[:, k],  "-",  label=f"a_des{labels[k]}",  color=colors[k])
            ax.plot(t, A_pred[:, k], "--", label=f"a_pred{labels[k]}", color=colors[k], alpha=0.9)
        ax.set_title("Linear Acceleration: (Λ⁻¹·wrench) vs ẍ_pred")
        ax.set_ylabel("m/s²"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        # (2) Linear: error + magnitude
        ax = axes[1, 0]
        for k in range(3):
            ax.plot(t, E_v[:, k], color=colors[k], label=f"e_a{labels[k]}")
        ax.plot(t, E_v_mag, "-", linewidth=2.0, color="black", label="‖e_a_v‖")
        ax.set_title("Linear Acceleration Error: (Λ⁻¹·wrench) − ẍ_pred")
        ax.set_ylabel("m/s²"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        # (3) Linear: magnitude only (zoomed view)
        ax = axes[2, 0]
        ax.plot(t, E_v_mag, linewidth=2.0)
        ax.set_title("‖Linear Error‖")
        ax.set_xlabel("time [s]"); ax.set_ylabel("m/s²"); ax.grid(True, alpha=0.3)

        # (4) Angular: a_des vs a_pred
        ax = axes[0, 1]
        for k in range(3):
            ax.plot(t, A_des[:, k+3],  "-",  label=f"α_des{labels[k]}",  color=colors[k])
            ax.plot(t, A_pred[:, k+3], "--", label=f"α_pred{labels[k]}", color=colors[k], alpha=0.9)
        ax.set_title("Angular Acceleration: (Λ⁻¹·wrench) vs ẍ_pred")
        ax.set_ylabel("rad/s²"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        # (5) Angular: error + magnitude
        ax = axes[1, 1]
        for k in range(3):
            ax.plot(t, E_w[:, k], color=colors[k], label=f"e_α{labels[k]}")
        ax.plot(t, E_w_mag, "-", linewidth=2.0, color="black", label="‖e_a_w‖")
        ax.set_title("Angular Acceleration Error: (Λ⁻¹·wrench) − ẍ_pred")
        ax.set_ylabel("rad/s²"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        # (6) Angular: magnitude only
        ax = axes[2, 1]
        ax.plot(t, E_w_mag, linewidth=2.0)
        ax.set_title("‖Angular Error‖")
        ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²"); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.035, 1, 0.97])
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    # impedance gains (example)
    K = np.diag([10, 10, 10, 0.1, 0.1, 0.1])
    robot = URImpedanceController("192.168.1.33", K=K)

    # diagonal task weighting
    W_imp = diag6([1, 1, 1, 1e2, 1e2, 1e2])

    traj_path = "motion_planner/trajectories_old/pick_and_place.npz"
    Hz = 50

    ctrl = SingleArmImpedanceQP(
        robot=robot,
        Hz=Hz,
        trajectory_npz_path=traj_path,
        W_imp=W_imp,
        lambda_reg=5e-8,
        tikhonov_lambda=1e-4,
    )

    try:
        robot.go_home()
        robot.wait_for_commands()
        robot.wait_until_done()
        robot.go_to_approach()
        robot.wait_for_commands()
        robot.wait_until_done()

        ctrl.run()

        # plots
        ctrl.plot_qp_objective()
        ctrl.plot_qp_performance()
        ctrl.plot_jointspace()
        ctrl.plot_taskspace()
        ctrl.plot_wrench_lambda_inv_minus_xddot()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        try:
            robot.stop_control()
        except Exception:
            pass
    except Exception as e:
        print(f"Error: {e}")
        try:
            robot.stop_control()
        except Exception:
            pass
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Robot disconnected.")
