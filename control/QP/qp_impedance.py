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
        grasping_point, _, self.normal = self.get_grasping_data()
        self.grasping_point = self.world_point_2_robot(grasping_point)

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

class DualArmImpedanceQP:
    def __init__(
        self,
        robot_L: URImpedanceController,
        robot_R: URImpedanceController,
        Hz: int,
        trajectory_npz_path: str,
        W_imp_L=None,
        W_imp_R=None,
        lambda_reg: float = 1e-6,
    ):
        self.robot_L = robot_L
        self.robot_R = robot_R
        self.Hz = int(Hz)
        self.dt = 1.0 / float(self.Hz)

        self._npz_path = trajectory_npz_path
        data = np.load(self._npz_path)
        self.traj_len = int(data["position"].shape[0])

        self.joint_pose_limit  = np.deg2rad(360.0)
        self.joint_speed_limit = np.deg2rad(180.0)
        self.joint_accel_limit = np.deg2rad(120.0)

        self.W_imp_L_np = np.asarray(W_imp_L, dtype=float) if W_imp_L is not None else np.eye(6)
        self.W_imp_R_np = np.asarray(W_imp_R, dtype=float) if W_imp_R is not None else np.eye(6)
        self.lambda_reg = float(lambda_reg)

        self.control_data = []
        self._should_stop = threading.Event()

        self.box_dimensions = np.linalg.norm(self.robot_L.get_grasping_data()[0] - self.robot_R.get_grasping_data()[0])
        self.normal_L = self.robot_L.get_grasping_data()[2]
        self.normal_R = self.robot_R.get_grasping_data()[2]

        self._build_qp(self.dt)


    def _load_refs(self, grasping_point_L, grasping_point_R, grasping_R_L, grasping_R_R):
        data  = np.load(self._npz_path)
        p_box = data["position"]            # (T,3) displacement of box center from t=0
        v_box = data["linear_velocity"]     # (T,3)
        R_box = data["rotation_matrices"]   # (T,3,3) absolute; R_box[0] is initial
        w_box = data["angular_velocity"]    # (T,3)
        self.traj_len = int(p_box.shape[0])

        # LEFT init
        p0_L, R0_L = grasping_point_L, grasping_R_L
        self.p_ref_L, self.v_ref_L = np.zeros_like(p_box), np.zeros_like(v_box)
        self.R_ref_L, self.w_ref_L = np.zeros_like(R_box), np.zeros_like(w_box)

        # Right
        p0_R, R0_R = grasping_point_R, grasping_R_R
        self.p_ref_R, self.v_ref_R = np.zeros_like(p_box), np.zeros_like(v_box)
        self.R_ref_R, self.w_ref_R = np.zeros_like(R_box), np.zeros_like(w_box)

        # Geometry: single side length (distance between the two faces being grasped)
        L = float(self.box_dimensions)
        R_box0 = R_box[0]  # world <- B0

        nL_W = np.array(self.normal_L, dtype=float)
        nR_W = np.array(self.normal_R, dtype=float)

        # Lever arms in the initial box frame (fixed in B0)
        rL_B0 =  L * (R_box0.T @ nL_W)
        rR_B0 = - L * (R_box0.T @ nR_W)

        # ---------- keep your loop shape ----------
        for t in range(self.traj_len):
            # Left
            self.p_ref_L[t] = p0_L + p_box[t]
            self.v_ref_L[t] = v_box[t]
            R_rel = R_box[t] @ R_box0.T
            self.R_ref_L[t] = R_rel @ R0_L
            self.w_ref_L[t] = w_box[t]
            # Right
            self.p_ref_R[t] = p0_R + p_box[t]
            self.v_ref_R[t] = v_box[t]
            self.R_ref_R[t] = R_rel @ R0_R
            self.w_ref_R[t] = w_box[t]

            # Universal 3D rotational offsets in world:
            # Δp_a^W(t) = (R_box[t] - R_box0) @ r_a^B0
            self.p_ref_L[t] += (R_box[t] - R_box0) @ rL_B0
            self.p_ref_R[t] += (R_box[t] - R_box0) @ rR_B0

    def _build_qp(self, dt):
        n = 6
        self.q_ddot_L = cp.Variable(n, name="q_ddot_L")
        self.q_ddot_R = cp.Variable(n, name="q_ddot_R")

        # Params Left
        self.J_L_p      = cp.Parameter((6, n), name="J_L")
        self.J_dot_L_p  = cp.Parameter((6, n), name="J_dot_L")
        self.q_L_p      = cp.Parameter(n,      name="q_L")
        self.q_dot_L_p  = cp.Parameter(n,      name="q_dot_L")
        self.a_des_L_p  = cp.Parameter(6,      name="a_des_L")
        self.W_imp_L_p  = cp.Parameter((6, 6), name="W_imp_L")

        # Params Right
        self.J_R_p      = cp.Parameter((6, n), name="J_R")
        self.J_dot_R_p  = cp.Parameter((6, n), name="J_dot_R")
        self.q_R_p      = cp.Parameter(n,      name="q_R")
        self.q_dot_R_p  = cp.Parameter(n,      name="q_dot_R")
        self.a_des_R_p  = cp.Parameter(6,      name="a_des_R")
        self.W_imp_R_p  = cp.Parameter((6, 6), name="W_imp_R")

        self.dt_c  = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(0.5 * dt * dt))

        # Left Dynamics
        q_next_L     = self.q_L_p     + self.q_dot_L_p * self.dt_c + self.q_ddot_L * self.dt2_c
        q_dot_next_L = self.q_dot_L_p + self.q_ddot_L * self.dt_c
        e_imp_L      = self.J_L_p @ self.q_ddot_L + self.J_dot_L_p @ self.q_dot_L_p - self.a_des_L_p

        # Right Dynamics
        q_next_R     = self.q_R_p     + self.q_dot_R_p * self.dt_c + self.q_ddot_R * self.dt2_c
        q_dot_next_R = self.q_dot_R_p + self.q_ddot_R * self.dt_c
        e_imp_R      = self.J_R_p @ self.q_ddot_R + self.J_dot_R_p @ self.q_dot_R_p - self.a_des_R_p

        obj = (cp.sum_squares(self.W_imp_L_p @ e_imp_L) + 
               cp.sum_squares(self.W_imp_R_p @ e_imp_R) + 
               self.lambda_reg * cp.sum_squares(self.q_ddot_L) + 
               self.lambda_reg * cp.sum_squares(self.q_ddot_R))

        cons = [
            -self.joint_pose_limit <= q_next_L,     q_next_L     <= self.joint_pose_limit,
            -self.joint_speed_limit <= q_dot_next_L, q_dot_next_L <= self.joint_speed_limit,
            -self.joint_accel_limit <= self.q_ddot_L,  self.q_ddot_L  <= self.joint_accel_limit,
            
            -self.joint_pose_limit <= q_next_R,     q_next_R     <= self.joint_pose_limit,
            -self.joint_speed_limit <= q_dot_next_R, q_dot_next_R <= self.joint_speed_limit,
            -self.joint_accel_limit <= self.q_ddot_R,  self.q_ddot_R  <= self.joint_accel_limit,
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.prob_kwargs = dict(eps_abs=1e-6, eps_rel=1e-6, alpha=1.6, max_iter=10000,
                                adaptive_rho=True, adaptive_rho_interval=45, polish=True,
                                check_termination=10, warm_start=True)

    def run(self):
        time.sleep(0.1)
        
        dt = self.dt
        t_idx = 0
        self._should_stop.clear()
        traj_initialized = False
        
        print("[DualArmQP] Starting Joint Impedance QP control loop")
        i = 0

        start_p_L = self.robot_L.get_state()["gripper_base"][:3]
        start_p_R = self.robot_R.get_state()["gripper_base"][:3]
        grasping_point_L = self.robot_L.grasping_point
        grasping_point_R = self.robot_R.grasping_point
        grasping_R_L = RR.from_rotvec(self.robot_L.get_state()["pose"][3:6]).as_matrix()
        grasping_R_R = RR.from_rotvec(self.robot_R.get_state()["pose"][3:6]).as_matrix()
        self._ctrl_start_wall = time.perf_counter()


        while not self._should_stop.is_set():
            loop_t0 = time.perf_counter()
            elapsed = loop_t0 - self._ctrl_start_wall
            try:
                if t_idx >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # --- State LEFT ---
                state_L = self.robot_L.get_state()
                p_L = np.array(state_L["gripper_base"][:3])
                v_L = np.array(state_L["speed"][:3])
                R_L = RR.from_rotvec(state_L["pose"][3:6]).as_matrix()
                w_L = np.array(state_L["speed"][3:6])
                q_L, q_dot_L = self.robot_L.get_q(), self.robot_L.get_qdot()
                
                J_L     = self.robot_L.get_J(q_L)
                J_dot_L = self.robot_L.get_Jdot(q_L, q_dot_L)
                M_L     = self.robot_L.get_M(q_L)
                Lam_L, Lam_inv_L = self.robot_L.get_Lambda_and_inv(J_L, M_L)
                D_L     = self.robot_L.get_D(self.robot_L.K, Lam_L)

                if elapsed < 4.0:
                    alpha = elapsed / 4.0
                    p_ref_L = start_p_L + alpha * (grasping_point_L - start_p_L)
                    R_ref_L = grasping_R_L
                    v_ref_L, w_ref_L = np.zeros(3), np.zeros(3)
                    
                    p_ref_R = start_p_R + alpha * (grasping_point_R - start_p_R)
                    R_ref_R = grasping_R_R
                    v_ref_R, w_ref_R = np.zeros(3), np.zeros(3)
                else:
                    if not traj_initialized:
                        print("Loading trajectory references...")
                        self._load_refs(grasping_point_L, grasping_point_R, grasping_R_L, grasping_R_R)
                        traj_initialized = True
                        t_idx = 0
                    
                    if t_idx < self.traj_len:
                        p_ref_L, v_ref_L = self.p_ref_L[t_idx], self.v_ref_L[t_idx]
                        R_ref_L, w_ref_L = self.R_ref_L[t_idx], self.w_ref_L[t_idx]
                        p_ref_R, v_ref_R = self.p_ref_R[t_idx], self.v_ref_R[t_idx]
                        R_ref_R, w_ref_R = self.R_ref_R[t_idx], self.w_ref_R[t_idx]
                        t_idx += 1

                e_p_L = p_ref_L - p_L
                e_v_L = v_ref_L - v_L
                e_r_L = short_arc_log(R_ref_L, R_L)
                e_w_L = w_ref_L - w_L
                f_des_L = self.robot_L.wrench_desired(self.robot_L.K, D_L, e_p_L, e_r_L, e_v_L, e_w_L)
                a_des_L = Lam_inv_L @ f_des_L

                # --- State RIGHT ---
                state_R = self.robot_R.get_state()
                p_R = np.array(state_R["gripper_base"][:3])
                v_R = np.array(state_R["speed"][:3])
                R_R = RR.from_rotvec(state_R["pose"][3:6]).as_matrix()
                w_R = np.array(state_R["speed"][3:6])
                q_R, q_dot_R = self.robot_R.get_q(), self.robot_R.get_qdot()

                J_R     = self.robot_R.get_J(q_R)
                J_dot_R = self.robot_R.get_Jdot(q_R, q_dot_R)
                M_R     = self.robot_R.get_M(q_R)
                Lam_R, Lam_inv_R = self.robot_R.get_Lambda_and_inv(J_R, M_R)
                D_R     = self.robot_R.get_D(self.robot_R.K, Lam_R)

                e_p_R = p_ref_R - p_R
                e_v_R = v_ref_R - v_R
                e_r_R = short_arc_log(R_ref_R, R_R)
                e_w_R = w_ref_R - w_R
                f_des_R = self.robot_R.wrench_desired(self.robot_R.K, D_R, e_p_R, e_r_R, e_v_R, e_w_R)
                a_des_R = Lam_inv_R @ f_des_R

                # --- Populate QP ---
                # Left
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.J_dot_L_p.value = _freeze_sparsity(J_dot_L)
                self.q_L_p.value = q_L
                self.q_dot_L_p.value = q_dot_L
                self.a_des_L_p.value = a_des_L
                self.W_imp_L_p.value = self.W_imp_L_np

                # Right
                self.J_R_p.value = _freeze_sparsity(J_R)
                self.J_dot_R_p.value = _freeze_sparsity(J_dot_R)
                self.q_R_p.value = q_R
                self.q_dot_R_p.value = q_dot_R
                self.a_des_R_p.value = a_des_R
                self.W_imp_R_p.value = self.W_imp_R_np

                # --- Solve ---
                t_solve = time.perf_counter()
                self.prob.solve(solver=cp.OSQP, **self.prob_kwargs)
                solve_dt = time.perf_counter() - t_solve

                status = (self.prob.status or "").lower()
                ok = status in ("optimal", "optimal_inaccurate")
                
                q_ddot_L_sol = _as_rowvec_1d(self.q_ddot_L.value, "q_ddot_L", 6) if ok else np.zeros(6)
                q_ddot_R_sol = _as_rowvec_1d(self.q_ddot_R.value, "q_ddot_R", 6) if ok else np.zeros(6)

                # --- Command ---
                cmd_L = q_dot_L + q_ddot_L_sol * dt
                cmd_R = q_dot_R + q_ddot_R_sol * dt
                self.robot_L.speedJ(cmd_L.tolist(), dt)
                self.robot_R.speedJ(cmd_R.tolist(), dt)

                # --- Metrics for Logging ---
                # Imp Left
                e_imp_L_val = J_L @ q_ddot_L_sol + J_dot_L @ q_dot_L - a_des_L
                imp_L_term = float((self.W_imp_L_np @ e_imp_L_val).T @ (self.W_imp_L_np @ e_imp_L_val))
                # Imp Right
                e_imp_R_val = J_R @ q_ddot_R_sol + J_dot_R @ q_dot_R - a_des_R
                imp_R_term = float((self.W_imp_R_np @ e_imp_R_val).T @ (self.W_imp_R_np @ e_imp_R_val))
                # Reg
                reg_term = float(self.lambda_reg) * (float(q_ddot_L_sol @ q_ddot_L_sol) + float(q_ddot_R_sol @ q_ddot_R_sol))
                obj_total = imp_L_term + imp_R_term + reg_term

                # --- Logging ---
                tcp_L = {
                    "p": p_L, "v": v_L, "w": w_L, "rvec": RR.from_matrix(R_L).as_rotvec(),
                    "p_ref": p_ref_L, "v_ref": v_ref_L, "w_ref": w_ref_L, "rvec_ref": RR.from_matrix(R_ref_L).as_rotvec(),
                    "e_p": e_p_L, "e_v": e_v_L, "e_w": e_w_L, "e_r": e_r_L
                }
                tcp_R = {
                    "p": p_R, "v": v_R, "w": w_R, "rvec": RR.from_matrix(R_R).as_rotvec(),
                    "p_ref": p_ref_R, "v_ref": v_ref_R, "w_ref": w_ref_R, "rvec_ref": RR.from_matrix(R_ref_R).as_rotvec(),
                    "e_p": e_p_R, "e_v": e_v_R, "e_w": e_w_R, "e_r": e_r_R
                }

                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": status,
                    "solver_time": solve_dt,
                    "obj_break": {"imp_L": imp_L_term, "imp_R": imp_R_term, "reg": reg_term, "total": obj_total},
                    "tcp_L": tcp_L,
                    "tcp_R": tcp_R,
                    "q_L": q_L, "q_dot_L": q_dot_L, "q_ddot_L": q_ddot_L_sol,
                    "q_R": q_R, "q_dot_R": q_dot_R, "q_ddot_R": q_ddot_R_sol,
                })

            except Exception as ex:
                print(f"Control Loop Error: {ex}")
                try:
                    self.robot_L.speedStop(); self.robot_R.speedStop()
                except: pass
                break

            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0, dt - elapsed))
            i += 1
            t_idx += 1

        try:
            self.robot_L.speedStop(); self.robot_R.speedStop()
        except: pass
        print(f"Done. Iterations: {i}")

    def plot_taskspace(self, arm: str, title_prefix="TaskspaceTracking"):
        """
        Plots task-space tracking performance for a specific arm.
        arm: "L" or "R"
        """
        if not self.control_data:
            print("No control_data to plot."); return

        # Select the correct key based on the arm argument
        key = f"tcp_{arm.upper()}"
        if key not in self.control_data[0]:
            print(f"Key {key} not found in control_data."); return

        os.makedirs("plots", exist_ok=True)
        
        # Time array
        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X", "Y", "Z"]

        # Extract data using the specific key (tcp_L or tcp_R)
        P        = np.array([d[key]["p"]        for d in self.control_data])
        V        = np.array([d[key]["v"]        for d in self.control_data])
        RVEC     = np.array([d[key]["rvec"]     for d in self.control_data])
        W        = np.array([d[key]["w"]        for d in self.control_data])
        P_ref    = np.array([d[key]["p_ref"]    for d in self.control_data])
        V_ref    = np.array([d[key]["v_ref"]    for d in self.control_data])
        RVEC_ref = np.array([d[key]["rvec_ref"] for d in self.control_data])
        W_ref    = np.array([d[key]["w_ref"]    for d in self.control_data])

        E_p = np.array([d[key]["e_p"] for d in self.control_data])
        E_v = np.array([d[key]["e_v"] for d in self.control_data])
        E_w = np.array([d[key]["e_w"] for d in self.control_data])
        E_r = np.array([d[key]["e_r"] for d in self.control_data])

        E_p_mag = np.linalg.norm(E_p, axis=1)
        E_v_mag = np.linalg.norm(E_v, axis=1)
        E_w_mag = np.linalg.norm(E_w, axis=1)
        E_r_mag = np.linalg.norm(E_r, axis=1)

        # Your exact plotting structure
        fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
        fig.suptitle(f"{title_prefix} Robot_{arm} – {timestamp_str}", fontsize=14, y=0.99)

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
        fname = f"plots/{title_prefix.lower()}_{arm}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()

    def plot_jointspace(self, arm: str, title_prefix="Joint_Space"):
        """
        Plots joint position, velocity, acceleration for a specific arm.
        arm: "L" or "R"
        """
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Select keys based on arm
        q_key = f"q_{arm.upper()}"
        qd_key = f"q_dot_{arm.upper()}"
        qdd_key = f"q_ddot_{arm.upper()}"

        q     = np.array([d[q_key] for d in self.control_data])
        qdot  = np.array([d[qd_key] for d in self.control_data])
        qddot = np.array([d[qdd_key] for d in self.control_data])
        
        n = q.shape[1] if q.ndim == 2 else 6
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"{title_prefix} Robot_{arm} – {timestamp_str}", fontsize=13, y=0.98)

        for j in range(n):
            axs[0].plot(t, q[:, j], label=f"q{j+1}")
            axs[1].plot(t, qdot[:, j], label=f"q̇{j+1}")
            axs[2].plot(t, qddot[:, j], label=f"q̈{j+1}")

        axs[0].set_ylabel("Position [rad]"); axs[0].legend(fontsize=7, ncol=3); axs[0].grid(True, alpha=0.3)
        axs[1].set_ylabel("Velocity [rad/s]"); axs[1].legend(fontsize=7, ncol=3); axs[1].grid(True, alpha=0.3)
        axs[2].set_ylabel("Acceleration [rad/s²]"); axs[2].legend(fontsize=7, ncol=3); axs[2].grid(True, alpha=0.3)
        axs[2].set_xlabel("Time [s]")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_{arm}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()

    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        """Objective-only: total vs impedance (L/R) vs regularizer."""
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        
        # Check for the breakdown dictionary
        if "obj_break" not in self.control_data[0]:
            print("No obj_break logged."); return

        # Extract breakdown data
        imp_L = np.array([d["obj_break"]["imp_L"] for d in self.control_data])
        imp_R = np.array([d["obj_break"]["imp_R"] for d in self.control_data])
        reg   = np.array([d["obj_break"]["reg"]   for d in self.control_data])
        tot   = np.array([d["obj_break"]["total"] for d in self.control_data])

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)
        ax.plot(t, tot, label="Total", linewidth=2, color="black")
        ax.plot(t, imp_L, label="Imp L", linestyle="--")
        ax.plot(t, imp_R, label="Imp R", linestyle="--")
        ax.plot(t, reg, label="Regularizer")
        
        ax.set_ylabel("Objective Value"); ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
        ax.set_title("QP Objective Breakdown", fontsize=11)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_qp_performance(self, title_prefix="QP_Performance"):
        """Plots solver timing statistics."""
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        
        # Extract solver time (convert to ms)
        solve_ms = np.array([d["solver_time"] for d in self.control_data]) * 1000.0
        
        # Calculate Rolling Frequency (Hz)
        loop_diff = np.diff(ts)
        # Prepend first dt or mean to keep array size same
        if len(loop_diff) > 0:
            loop_diff = np.insert(loop_diff, 0, loop_diff[0])
        else:
            loop_diff = np.zeros_like(ts)
            
        loop_hz = 1.0 / np.maximum(loop_diff, 1e-9)
        
        # Simple rolling mean for smoother Hz plot
        def rolling_mean(a, w=10):
            if len(a) < w: return a
            return np.convolve(a, np.ones(w)/w, mode='same')
            
        loop_hz_smooth = rolling_mean(loop_hz)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        # Solver Time
        axes[0].plot(t, solve_ms, label="Solver Time", color="orange")
        axes[0].axhline(y=20.0, color='r', linestyle='--', alpha=0.5, label="20ms limit (50Hz)")
        axes[0].set_ylabel("Time [ms]")
        axes[0].set_title("Solver Duration per Iteration")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="upper right")

        # Loop Frequency
        axes[1].plot(t, loop_hz_smooth, label="Loop Hz (Smoothed)", color="blue")
        axes[1].axhline(y=self.Hz, color='k', linestyle='--', alpha=0.5, label=f"Target {self.Hz}Hz")
        axes[1].set_ylabel("Frequency [Hz]")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_title("Control Loop Frequency")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right")

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    # Stiffness
    K = np.diag([10, 10, 10, 0.1, 0.1, 0.1])
    
    # Initialize Robots
    robot_L = URImpedanceController("192.168.1.33", K=K)
    robot_R = URImpedanceController("192.168.1.66", K=K)

    # Weights
    W_imp = diag6([1.0, 1.0, 1.0, 0.6, 0.6, 0.6])
    lambda_reg = 1e-6

    traj_path = "motion_planner/trajectories_old/twist.npz"
    Hz = 50

    ctrl = DualArmImpedanceQP(
        robot_L=robot_L,
        robot_R=robot_R,
        Hz=Hz,
        trajectory_npz_path=traj_path,
        W_imp_L=W_imp,
        W_imp_R=W_imp,
        lambda_reg=lambda_reg
    )

    try:
        robot_L.wait_for_commands()
        robot_L.moveJ(
            [-2.72771532, -1.40769446, 2.81887228, -3.01955523, -1.6224683, 2.31350756]
        )
        robot_L.wait_for_commands()

        print("Moving to Home...")
        robot_L.go_home(); robot_R.go_home()
        robot_L.wait_for_commands(); robot_R.wait_for_commands()
        robot_L.wait_until_done(); robot_R.wait_until_done()
        
        print("Moving to Approach...")
        robot_L.go_to_approach(); robot_R.go_to_approach()
        robot_L.wait_for_commands(); robot_R.wait_for_commands()
        robot_L.wait_until_done(); robot_R.wait_until_done()

        ctrl.run()
        
        ctrl.plot_taskspace("L")
        ctrl.plot_jointspace("L")
        ctrl.plot_taskspace("R")
        ctrl.plot_jointspace("R")
        
        ctrl.plot_qp_objective()
        ctrl.plot_qp_performance()

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        try:
            robot_L.disconnect(); robot_R.disconnect()
        except: pass
        print("Robots disconnected.")