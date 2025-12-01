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
from utils.utils import _freeze_sparsity, _as_rowvec_1d, short_arc_log, diag6


# ==============================
#   UR Impedance Controller
#   (unchanged; keep for ref-tracking)
# ==============================
class URImpedanceController(URForceController):
    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = np.asarray(K, dtype=float)
        self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("tool0")

    def get_q(self):
        return np.array(self.rtde_receive.getActualQ(), dtype=float)

    def get_qdot(self):
        return np.array(self.rtde_receive.getActualQd(), dtype=float)

    # --- UR-side dynamics (optional; kept) ---
    def get_J_ur(self, q):
        J_flat = self.rtde_control.getJacobian(q.tolist())
        return np.array(J_flat, dtype=float).reshape((6, 6))

    def get_Jdot_ur(self, q, v):
        Jdot_flat = self.rtde_control.getJacobianTimeDerivative(q.tolist(), v.tolist())
        return np.array(Jdot_flat, dtype=float).reshape((6, 6))

    def get_M_ur(self, q):
        M_flat = self.rtde_control.getMassMatrix(q.tolist(), include_rotors_inertia=True)
        M = np.array(M_flat, dtype=float).reshape((6, 6))
        return 0.5 * (M + M.T)
    
    # --- Pinocchio Jacobians/dynamics (used) ---
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

    # --- Impedance wrench<->accel mapping ---
    def get_Lambda_and_inv(self, J, M, rel_reg=1e-3, abs_reg=1e-3):
        X = solve(M, J.T)
        A = J @ X
        A = 0.5 * (A + A.T)
        eigvals, Q = eigh(A)
        lam_max = float(np.max(eigvals))
        floor = max(abs_reg, rel_reg * lam_max)
        eigvals_reg = np.maximum(eigvals, floor)
        
        Lambda_inv = (Q * eigvals_reg) @ Q.T
        Lambda_inv = 0.5 * (Lambda_inv + Lambda_inv.T)

        inv_eigs = 1.0 / eigvals_reg
        Lambda = (Q * inv_eigs) @ Q.T
        Lambda = 0.5 * (Lambda + Lambda.T)
        return Lambda, Lambda_inv
    
    def get_D(self, K, Lambda):
        K = 0.5 * (K + K.T)
        L = 0.5 * (Lambda + Lambda.T)
        wK, VK = np.linalg.eigh(K)
        wK = np.maximum(wK, 1e-12)
        S_K = (VK * np.sqrt(wK)) @ VK.T
        S_K = 0.5 * (S_K + S_K.T)

        wL, VL = np.linalg.eigh(L)
        wL = np.maximum(wL, 1e-12)
        S_L = (VL * np.sqrt(wL)) @ VL.T
        S_L = 0.5 * (S_L + S_L.T)

        D = S_L @ S_K + S_K @ S_L
        D = 0.5 * (D + D.T)
        return D
    
    def wrench_desired(self, K, D, e_p, e_r, e_v, e_w):
        X_err = np.hstack((e_p, e_r))
        Xd_err = np.hstack((e_v, e_w))
        return D @ Xd_err + K @ X_err


# ================================================================
# Dual-arm controller: keep impedance tracking,
# swap grasp admittance with accel->vel mass–spring–damper
# ================================================================
class DualArmImpedanceAdmittanceQP:
    def __init__(
        self,
        robot_L: URImpedanceController,
        robot_R: URImpedanceController,
        Hz: int,
        trajectory_npz_path: str,
        # weights for impedance (same)
        W_imp_L=None,
        W_imp_R=None,
        # weights for admittance velocity tracking (re-using grasp weight slots)
        W_grasp_L=None,
        W_grasp_R=None,
        lambda_reg: float = 1e-6,
        # --- NEW: admittance (1D along grasp normal) params ---
        M_a: float = 3.0,
        D_a: float = 40.0,
        K_a: float = 0.0,
        v_max: float = 0.05,
        ref_force: float = 20.0,
    ):
        self.robotL = robot_L
        self.robotR = robot_R
        self.Hz = int(Hz)
        self.dt = 1.0 / float(self.Hz)

        # --- Admittance parameters (shared per-arm; 1D along normal) ---
        self.M_a_L = float(M_a); self.D_a_L = float(D_a); self.K_a_L = float(K_a)
        self.M_a_R = float(M_a); self.D_a_R = float(D_a); self.K_a_R = float(K_a)
        self.v_max = float(v_max)
        self.ref_force = float(ref_force)

        self._npz_path = trajectory_npz_path
        self.data = np.load(self._npz_path)
        self.traj_len = int(self.data["position"].shape[0])

        # Limits
        self.joint_pose_limit  = np.deg2rad(360.0)
        self.joint_speed_limit = np.deg2rad(180.0)
        self.joint_accel_limit = np.deg2rad(120.0)

        # QP Weights
        self.W_imp_L = np.asarray(W_imp_L, dtype=float) if W_imp_L is not None else np.eye(6)
        self.W_imp_R = np.asarray(W_imp_R, dtype=float) if W_imp_R is not None else np.eye(6)
        # interpret these as "velocity tracking" (admittance) weights
        self.W_grasp_L_np = np.asarray(W_grasp_L, dtype=float) if W_grasp_L is not None else np.zeros((6,6))
        self.W_grasp_R_np = np.asarray(W_grasp_R, dtype=float) if W_grasp_R is not None else np.zeros((6,6))
        self.lambda_reg = float(lambda_reg)

        self.control_data = []
        self._should_stop = threading.Event()
        
        # Perf metrics
        self.log_every_s = 1.0
        self._ctrl_start_wall = None
        self._last_log_wall = None
        self._win_iters = 0
        self._win_loop_time = 0.0
        self._win_solver_time = 0.0
        self._win_deadline_miss = 0
        self._total_iters = 0
        self._total_solver_time = 0.0
        self._total_deadline_miss = 0

        # Admittance states (normal direction)
        self.x_n_L = 0.0; self.v_n_L = 0.0
        self.x_n_R = 0.0; self.v_n_R = 0.0

        self.build_qp(self.dt)

    # --------- helper: grasp points + normals (BASE frame) ----------
    def _init_grasping_data(self):
        self.grasping_point_L_world, _, self.normal_L_world = self.robotL.get_grasping_data()
        self.grasping_point_L_base = self.robotL.world_point_2_robot(self.grasping_point_L_world)
        self.normal_L_base = self.robotL.world_vector_2_robot(self.normal_L_world)

        self.grasping_point_R_world, _, self.normal_R_world = self.robotR.get_grasping_data()
        self.grasping_point_R_base = self.robotR.world_point_2_robot(self.grasping_point_R_world)
        self.normal_R_base = self.robotR.world_vector_2_robot(self.normal_R_world)

        # normalize for safety
        nL = np.linalg.norm(self.normal_L_base); nR = np.linalg.norm(self.normal_R_base)
        if nL > 1e-12: self.normal_L_base = self.normal_L_base / nL
        if nR > 1e-12: self.normal_R_base = self.normal_R_base / nR

        # box dimension used before (kept if needed elsewhere)
        self.box_dimensions = np.linalg.norm(self.grasping_point_L_base - self.grasping_point_R_base)

    # --------- load reference trajectory for impedance (unchanged) ----------
    def _load_refs(self, grasping_point_L, grasping_point_R, grasping_R_L, grasping_R_R):
        p_box = self.data["position"]            # (T,3) displacement of box center from t=0
        v_box = self.data["linear_velocity"]     # (T,3)
        R_box = self.data["rotation_matrices"]   # (T,3,3) absolute; R_box[0] is initial
        w_box = self.data["angular_velocity"]    # (T,3)

        # LEFT init
        p0_L, R0_L = grasping_point_L, grasping_R_L
        self.p_ref_L, self.v_ref_L = np.zeros_like(p_box), np.zeros_like(v_box)
        self.R_ref_L, self.w_ref_L = np.zeros_like(R_box), np.zeros_like(w_box)

        # RIGHT init
        p0_R, R0_R = grasping_point_R, grasping_R_R
        self.p_ref_R, self.v_ref_R = np.zeros_like(p_box), np.zeros_like(v_box)
        self.R_ref_R, self.w_ref_R = np.zeros_like(R_box), np.zeros_like(w_box)
        
        R_box0 = R_box[0]
        for t in range(self.traj_len):
            # Left (same as before)
            self.p_ref_L[t] = p0_L + p_box[t]
            self.v_ref_L[t] = v_box[t]
            self.R_ref_L[t] = (R_box[t] @ R_box0.T) @ R0_L
            self.w_ref_L[t] = w_box[t]
            # Right (mirror x only, like before)
            self.p_ref_R[t] = p0_R + p_box[t] * [-1, 1, 1]
            self.v_ref_R[t] = v_box[t] * [-1, 1, 1]
            self.R_ref_R[t] = (R_box[t] @ R_box0.T) @ R0_R
            self.w_ref_R[t] = w_box[t] 

    # --------- QP (unchanged form): accel var + impedance accel + admittance vel ----------
    def build_qp(self, dt):
        n = 6
        self.qddot_L = cp.Variable(n, name="qddot_L")
        self.qddot_R = cp.Variable(n, name="qddot_R")

        # --- Parameters ---
        self.J_L_p = cp.Parameter((6, n), name="J_L")
        self.J_dot_L_p = cp.Parameter((6, n), name="J_dot_L")
        self.q_L_p = cp.Parameter(n, name="q_L")
        self.qdot_L_p = cp.Parameter(n, name="qdot_L")
        
        # Targets
        self.a_des_L_p = cp.Parameter(6, name="a_des_L")        # Impedance Target (Accel)
        self.xdot_star_L = cp.Parameter(6, name="xdot_star_L")  # Admittance Target (Vel)
        
        self.J_R_p = cp.Parameter((6, n), name="J_R")
        self.J_dot_R_p = cp.Parameter((6, n), name="J_dot_R")
        self.q_R_p = cp.Parameter(n, name="q_R")
        self.qdot_R_p = cp.Parameter(n, name="qdot_R")
        
        self.a_des_R_p = cp.Parameter(6, name="a_des_R")
        self.xdot_star_R = cp.Parameter(6, name="xdot_star_R")

        # Weights
        self.W_imp_L_p = cp.Parameter((6, 6), name="W_imp_L")
        self.W_grasp_L_p = cp.Parameter((6, 6), name="W_grasp_L")
        self.W_imp_R_p = cp.Parameter((6, 6), name="W_imp_R")
        self.W_grasp_R_p = cp.Parameter((6, 6), name="W_grasp_R")

        dt_c  = cp.Constant(float(dt))
        dt2_c = cp.Constant(float(0.5 * dt * dt))

        # --- Next-step states ---
        q_next_L     = self.q_L_p + self.qdot_L_p * dt_c + self.qddot_L * dt2_c
        q_dot_next_L = self.qdot_L_p + self.qddot_L * dt_c
        
        q_next_R     = self.q_R_p + self.qdot_R_p * dt_c + self.qddot_R * dt2_c
        q_dot_next_R = self.qdot_R_p + self.qddot_R * dt_c

        # --- Objectives ---
        # Impedance: match desired accel
        e_imp_L = self.J_L_p @ self.qddot_L + self.J_dot_L_p @ self.qdot_L_p - self.a_des_L_p
        e_imp_R = self.J_R_p @ self.qddot_R + self.J_dot_R_p @ self.qdot_R_p - self.a_des_R_p
        
        # Admittance: match desired twist velocity at next step
        x_dot_next_L = self.J_L_p @ q_dot_next_L
        e_grasp_L    = x_dot_next_L - self.xdot_star_L

        x_dot_next_R = self.J_R_p @ q_dot_next_R
        e_grasp_R    = x_dot_next_R - self.xdot_star_R

        # Combined Cost
        obj_L = cp.sum_squares(self.W_imp_L_p @ e_imp_L) + cp.sum_squares(self.W_grasp_L_p @ e_grasp_L)
        obj_R = cp.sum_squares(self.W_imp_R_p @ e_imp_R) + cp.sum_squares(self.W_grasp_R_p @ e_grasp_R)
        reg   = self.lambda_reg * (cp.sum_squares(self.qddot_L) + cp.sum_squares(self.qddot_R))
        obj = obj_L + obj_R + reg

        cons = [
            -self.joint_pose_limit <= q_next_L,     q_next_L     <= self.joint_pose_limit,
            -self.joint_pose_limit <= q_next_R,     q_next_R     <= self.joint_pose_limit,
            -self.joint_speed_limit <= q_dot_next_L, q_dot_next_L <= self.joint_speed_limit,
            -self.joint_speed_limit <= q_dot_next_R, q_dot_next_R <= self.joint_speed_limit,
            -self.joint_accel_limit <= self.qddot_L, self.qddot_L <= self.joint_accel_limit,
            -self.joint_accel_limit <= self.qddot_R, self.qddot_R <= self.joint_accel_limit,
        ]
        
        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(eps_abs=1e-6, eps_rel=1e-6, alpha=1.6, max_iter=10000,
                              adaptive_rho=True, adaptive_rho_interval=20, polish=True,
                              check_termination=10, warm_start=True)

    # =====================
    #         RUN
    # =====================
    def run(self):
        time.sleep(0.5)
        dt = 1.0 / self.Hz
        self.control_stop = threading.Event()
        i = 0
        t_idx = 0
        traj_initialized = False

        # Init grasp data/normals
        self._init_grasping_data()

        # Reset admittance states
        self.x_n_L = 0.0; self.v_n_L = 0.0
        self.x_n_R = 0.0; self.v_n_R = 0.0

        # Init start positions for traj interp
        start_p_L = self.robotL.get_state()["gripper_base"][:3]
        start_p_R = self.robotR.get_state()["gripper_base"][:3]
        grasping_R_L = RR.from_rotvec(self.robotL.get_state()["pose"][3:6]).as_matrix()
        grasping_R_R = RR.from_rotvec(self.robotR.get_state()["pose"][3:6]).as_matrix()

        # zero FT (optional; keep behavior consistent with new controller)
        try:
            self.robotL.rtde_control.zeroFtSensor()
            self.robotR.rtde_control.zeroFtSensor()
        except Exception:
            pass

        self._ctrl_start_wall = time.perf_counter()
        self._last_log_wall = time.perf_counter()
        
        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - self._ctrl_start_wall
            try:
                if t_idx >= self.traj_len:
                    print("Trajectory completed.")
                    break

                # --- state LEFT ---
                state_L = self.robotL.get_state()
                p_L, v_L = np.array(state_L["gripper_base"][:3]), np.array(state_L["speed"][:3])
                R_L = RR.from_rotvec(state_L["pose"][3:6]).as_matrix()
                w_L = np.array(state_L["speed"][3:6])
                q_L, qdot_L = self.robotL.get_q(), self.robotL.get_qdot()
                n_L = self.normal_L_base
                F_L = np.array(state_L["filtered_force"][:3])
                J_L = self.robotL.get_J(q_L)
                J_dot_L = self.robotL.get_Jdot(q_L, qdot_L)

                # --- state RIGHT ---
                state_R = self.robotR.get_state()
                p_R, v_R = np.array(state_R["gripper_base"][:3]), np.array(state_R["speed"][:3])
                R_R = RR.from_rotvec(state_R["pose"][3:6]).as_matrix()
                w_R = np.array(state_R["speed"][3:6])
                q_R, qdot_R = self.robotR.get_q(), self.robotR.get_qdot()
                n_R = self.normal_R_base
                F_R = np.array(state_R["filtered_force"][:3])
                J_R = self.robotR.get_J(q_R)
                J_dot_R = self.robotR.get_Jdot(q_R, qdot_R)

                # ==========================================================
                # 1) IMPEDANCE: trajectory tracking (unchanged)
                # ==========================================================
                t_model_start = time.perf_counter()
                
                if elapsed < 5:
                    alpha = elapsed / 5
                    p_ref_L = start_p_L + alpha * (self.grasping_point_L_base - start_p_L)
                    R_ref_L = grasping_R_L
                    v_ref_L, w_ref_L = np.zeros(3), np.zeros(3)
                    
                    p_ref_R = start_p_R + alpha * (self.grasping_point_R_base - start_p_R)
                    R_ref_R = grasping_R_R
                    v_ref_R, w_ref_R = np.zeros(3), np.zeros(3)
                else:
                    if not traj_initialized:
                        self._load_refs(self.grasping_point_L_base, self.grasping_point_R_base, grasping_R_L, grasping_R_R)
                        traj_initialized = True
                        t_idx = 0
                    
                    if t_idx < self.traj_len:
                        p_ref_L, v_ref_L = self.p_ref_L[t_idx], self.v_ref_L[t_idx]
                        R_ref_L, w_ref_L = self.R_ref_L[t_idx], self.w_ref_L[t_idx]
                        p_ref_R, v_ref_R = self.p_ref_R[t_idx], self.v_ref_R[t_idx]
                        R_ref_R, w_ref_R = self.R_ref_R[t_idx], self.w_ref_R[t_idx]
                        t_idx += 1
                
                # Desired Wrench -> Acceleration (Left)
                M_L = self.robotL.get_M(q_L)
                Lam_L, Lam_inv_L = self.robotL.get_Lambda_and_inv(J_L, M_L)
                D_L = self.robotL.get_D(self.robotL.K, Lam_L)
                e_p_L, e_v_L = p_ref_L - p_L, v_ref_L - v_L
                e_r_L, e_w_L = short_arc_log(R_ref_L, R_L), w_ref_L - w_L
                f_des_L = self.robotL.wrench_desired(self.robotL.K, D_L, e_p_L, e_r_L, e_v_L, e_w_L)
                a_des_L = Lam_inv_L @ f_des_L

                # Desired Wrench -> Acceleration (Right)
                M_R = self.robotR.get_M(q_R)
                Lam_R, Lam_inv_R = self.robotR.get_Lambda_and_inv(J_R, M_R)
                D_R = self.robotR.get_D(self.robotR.K, Lam_R)
                e_p_R, e_v_R = p_ref_R - p_R, v_ref_R - v_R
                e_r_R, e_w_R = short_arc_log(R_ref_R, R_R), w_ref_R - w_R
                f_des_R = self.robotR.wrench_desired(self.robotR.K, D_R, e_p_R, e_r_R, e_v_R, e_w_R)
                a_des_R = Lam_inv_R @ f_des_R

                # ==========================================================
                # 2) ADMITTANCE (REPLACED): 1D MSD along grasp normals
                #     -> produce desired TCP linear velocities
                # ==========================================================
                # Reference normal forces
                F_n_star = self.ref_force
                F_n_L = float(n_L @ F_L)
                F_n_R = float(n_R @ F_R)
                e_n_L = F_n_star - F_n_L
                e_n_R = F_n_star - F_n_R

                # signed distance from grasp plane along +n (project TCP - grasp)
                d_L = float((p_L - p_ref_L) @ n_L)
                d_R = float((p_R - p_ref_R) @ n_R)

                # Integrate normal-state acceleration -> velocity (clip) ; x_n is the geometric distance
                a_n_L = (e_n_L - self.D_a_L * self.v_n_L - self.K_a_L * d_L) / max(self.M_a_L, 1e-9)
                a_n_R = (e_n_R - self.D_a_R * self.v_n_R - self.K_a_R * d_R) / max(self.M_a_R, 1e-9)

                self.v_n_L = np.clip(self.v_n_L + a_n_L * dt, -self.v_max, self.v_max)
                self.v_n_R = np.clip(self.v_n_R + a_n_R * dt, -self.v_max, self.v_max)
                self.x_n_L = d_L
                self.x_n_R = d_R

                # Desired TCP linear velocities along -n (push in when force too low)
                v_star_L = ( self.v_n_L) * (-n_L)
                v_star_R = ( self.v_n_R) * (-n_R)

                xdot_star_L = np.hstack([v_star_L, np.zeros(3)])
                xdot_star_R = np.hstack([v_star_R, np.zeros(3)])
                
                t_model_total = time.perf_counter() - t_model_start

                # --- feed QP ---
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.J_dot_L_p.value = _freeze_sparsity(J_dot_L)
                self.q_L_p.value = q_L
                self.qdot_L_p.value = qdot_L
                self.a_des_L_p.value = a_des_L
                self.xdot_star_L.value = xdot_star_L
                self.W_imp_L_p.value = self.W_imp_L
                self.W_grasp_L_p.value = self.W_grasp_L_np

                self.J_R_p.value = _freeze_sparsity(J_R)
                self.J_dot_R_p.value = _freeze_sparsity(J_dot_R)
                self.q_R_p.value = q_R
                self.qdot_R_p.value = qdot_R
                self.a_des_R_p.value = a_des_R
                self.xdot_star_R.value = xdot_star_R
                self.W_imp_R_p.value = self.W_imp_R
                self.W_grasp_R_p.value = self.W_grasp_R_np

                # --- solve ---
                _t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                solve_dt = time.perf_counter() - _t0
                self._win_solver_time += solve_dt
                self._total_solver_time += solve_dt

                status = (self.qp.status or "").lower()
                ok = status in ("optimal", "optimal_inaccurate")

                # extract accelerations; integrate to velocity command
                if ok:
                    qddot_L_cmd = _as_rowvec_1d(self.qddot_L.value, "qddot_L", 6)
                    qddot_R_cmd = _as_rowvec_1d(self.qddot_R.value, "qddot_R", 6)
                    qdot_L_cmd = qdot_L + qddot_L_cmd * dt
                    qdot_R_cmd = qdot_R + qddot_R_cmd * dt
                else:
                    qddot_L_cmd = np.zeros(6)
                    qddot_R_cmd = np.zeros(6)
                    qdot_L_cmd = qdot_L  # hold measured
                    qdot_R_cmd = qdot_R

                # --- send (UR speedJ expects target joint velocities) ---
                self.robotL.speedJ(qdot_L_cmd.tolist(), dt)
                self.robotR.speedJ(qdot_R_cmd.tolist(), dt)

                # --- Logging (SUPerset: supports both old & new plotting) ---
                # Original obj breakdown terms
                e_imp_L_val = J_L @ qddot_L_cmd + J_dot_L @ qdot_L - a_des_L
                imp_L_term = float((self.W_imp_L @ e_imp_L_val).T @ (self.W_imp_L @ e_imp_L_val))
                
                x_dot_next_L = J_L @ (qdot_L + qddot_L_cmd * dt)
                e_grasp_L_val = x_dot_next_L - xdot_star_L
                grasp_L_term = float((self.W_grasp_L_np @ e_grasp_L_val).T @ (self.W_grasp_L_np @ e_grasp_L_val))
                
                e_imp_R_val = J_R @ qddot_R_cmd + J_dot_R @ qdot_R - a_des_R
                imp_R_term = float((self.W_imp_R @ e_imp_R_val).T @ (self.W_imp_R @ e_imp_R_val))
                x_dot_next_R = J_R @ (qdot_R + qddot_R_cmd * dt)
                e_grasp_R_val = x_dot_next_R - xdot_star_R
                grasp_R_term = float((self.W_grasp_R_np @ e_grasp_R_val).T @ (self.W_grasp_R_np @ e_grasp_R_val))
                
                reg_term = float(self.lambda_reg) * (float(qddot_L_cmd @ qddot_L_cmd) + float(qddot_R_cmd @ qddot_R_cmd))
                obj_total = imp_L_term + imp_R_term + grasp_L_term + grasp_R_term + reg_term

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

                # Old plotting dict
                force_data_dict = {
                    "F_L_vec": F_L,      "F_R_vec": F_R,
                    "F_L_ref": self.ref_force * n_L,  "F_R_ref": self.ref_force * n_R,
                    "F_n_star": self.ref_force,
                    "v_star_L": v_star_L, "v_star_R": v_star_R, 
                    "F_n_L": F_n_L,      "F_n_R": F_n_R,
                }

                # Append superset log
                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": status,
                    "solver_time": solve_dt,
                    "model_time": t_model_total,
                    "obj_break": {
                        "imp_L": imp_L_term, "imp_R": imp_R_term,
                        "grasp_L": grasp_L_term, "grasp_R": grasp_R_term,
                        "reg": reg_term, "total": obj_total
                    },
                    "force_data": force_data_dict,  # old plots use this

                    # --- NEW admittance-friendly top-level keys (for new plots) ---
                    "F_n_L": F_n_L,
                    "F_n_R": F_n_R,
                    "ref_force": self.ref_force,
                    "x_n_L": self.x_n_L,
                    "x_n_R": self.x_n_R,
                    "v_n_L": self.v_n_L,
                    "v_n_R": self.v_n_R,
                    "a_n_L": a_n_L,
                    "a_n_R": a_n_R,

                    # joint
                    "q_L": q_L, "q_dot_L": qdot_L, "q_ddot_L": qddot_L_cmd,
                    "q_R": q_R, "q_dot_R": qdot_R, "q_ddot_R": qddot_R_cmd,

                    # base-frame metrics for new plotter
                    "p_L": p_L,
                    "p_R": p_R,
                    "grasp_L": self.grasping_point_L_base,
                    "grasp_R": self.grasping_point_R_base,
                    "d_L": d_L,
                    "d_R": d_R,
                    "v_tcp_L": np.hstack([v_L, w_L]),
                    "v_tcp_R": np.hstack([v_R, w_R]),
                    "v_des_L": xdot_star_L,
                    "v_des_R": xdot_star_R,

                    # tcp bundles for old taskspace plots
                    "tcp_L": tcp_L,
                    "tcp_R": tcp_R,
                })

                # --- perf ---
                elapsed_loop = time.perf_counter() - loop_start
                self._win_loop_time += elapsed_loop
                self._win_iters += 1
                self._total_iters += 1
                if elapsed_loop > dt + 1e-4:
                    self._win_deadline_miss += 1
                    self._total_deadline_miss += 1

                now = time.perf_counter()
                if now - self._last_log_wall >= self.log_every_s and self._win_iters > 0:
                    avg_period = self._win_loop_time / self._win_iters
                    avg_hz = 1.0 / avg_period if avg_period > 0 else np.nan
                    avg_solver_ms = (self._win_solver_time / self._win_iters) * 1000.0
                    miss_pct = 100.0 * self._win_deadline_miss / self._win_iters
                    print(f"[GRASP DUAL a-OPT] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms | "
                          f"miss {miss_pct:4.1f}% | F_L={F_n_L:6.2f} F_R={F_n_R:6.2f}")
                    self._win_iters = 0
                    self._win_loop_time = 0.0
                    self._win_solver_time = 0.0
                    self._win_deadline_miss = 0
                    self._last_log_wall = now

                time.sleep(max(0, dt - elapsed_loop))
                i += 1

            except KeyboardInterrupt:
                print("User stop.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                break

        self.robotL.speedStop()
        self.robotR.speedStop()
        total_time = time.perf_counter() - self._ctrl_start_wall
        print(f"[GRASP DUAL a-OPT SUMMARY] Ran {self._total_iters} iters @ "
              f"{self._total_iters/total_time:.1f} Hz")

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
        """Objective-only: total vs impedance (L/R) vs admittance (L/R) vs regularizer."""
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
        
        # Admittance (Grasp) terms
        adm_L = np.array([d["obj_break"]["grasp_L"] for d in self.control_data])
        adm_R = np.array([d["obj_break"]["grasp_R"] for d in self.control_data])
        
        reg   = np.array([d["obj_break"]["reg"]   for d in self.control_data])
        tot   = np.array([d["obj_break"]["total"] for d in self.control_data])

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        # Plot Total
        ax.plot(t, tot, label="Total", linewidth=2.5, color="black", alpha=0.8)
        
        # Plot Impedance (Solid Lines)
        ax.plot(t, imp_L, label="Imp L", linestyle="-", linewidth=1.5, alpha=0.8)
        ax.plot(t, imp_R, label="Imp R", linestyle="-", linewidth=1.5, alpha=0.8)
        
        # Plot Admittance (Dashed Lines)
        ax.plot(t, adm_L, label="Adm L", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.plot(t, adm_R, label="Adm R", linestyle="--", linewidth=1.5, alpha=0.8)
        
        # Plot Regularizer (Dotted Line)
        ax.plot(t, reg, label="Regularizer", linestyle=":", color="gray", linewidth=1.5)
        
        ax.set_ylabel("Objective Value"); ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        # Use 2 columns for the legend to make it cleaner
        ax.legend(loc="best", fontsize=9, ncol=3)
        ax.set_title("QP Objective Component Breakdown", fontsize=11)

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

    def plot_force_profile(self, title_prefix="DualAdmittanceAccel2Vel"):
        import os
        from datetime import datetime
        import numpy as np
        import matplotlib.pyplot as plt

        if not self.control_data:
            print("[plot_force_profile] No data to plot.")
            return

        data = self.control_data

        # time axis (relative)
        t0 = data[0]["t"]
        t = np.array([d["t"] - t0 for d in data])

        F_n_L = np.array([d["F_n_L"] for d in data])
        F_n_R = np.array([d["F_n_R"] for d in data])
        F_ref = np.array([d["ref_force"] for d in data])

        x_n_L = np.array([d["x_n_L"] for d in data])
        x_n_R = np.array([d["x_n_R"] for d in data])

        v_n_L = np.array([d["v_n_L"] for d in data])
        v_n_R = np.array([d["v_n_R"] for d in data])

        # --- FIGURE 1: normal-direction metrics ---
        fig1, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig1.suptitle(f"{title_prefix} – Normal / Admittance Metrics")

        # Forces along normal
        ax = axes[0]
        ax.plot(t, F_n_L, label="F_n_L")
        ax.plot(t, F_n_R, label="F_n_R")
        ax.plot(t, F_ref, "--", label="F_ref")
        ax.set_ylabel("Normal force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Distance along normal (x_n)
        ax = axes[1]
        ax.plot(t, x_n_L, label="x_n_L (d_L)")
        ax.plot(t, x_n_R, label="x_n_R (d_R)")
        ax.set_ylabel("Distance along -n [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Normal velocity (admittance state)
        ax = axes[2]
        ax.plot(t, v_n_L, label="v_n_L")
        ax.plot(t, v_n_R, label="v_n_R")
        ax.set_ylabel("v_n [m/s]")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig1.tight_layout(rect=[0, 0.03, 1, 0.97])

        # --- FIGURE 2: base-frame metrics (poses + velocity commands) ---
        # extract base-frame stuff
        p_L = np.stack([d["p_L"] for d in data], axis=0)        # (N,3)
        p_R = np.stack([d["p_R"] for d in data], axis=0)
        grasp_L = np.stack([d["grasp_L"] for d in data], axis=0)
        grasp_R = np.stack([d["grasp_R"] for d in data], axis=0)

        d_L = np.array([d["d_L"] for d in data])
        d_R = np.array([d["d_R"] for d in data])

        v_tcp_L = np.stack([d["v_tcp_L"] for d in data], axis=0)
        v_tcp_R = np.stack([d["v_tcp_R"] for d in data], axis=0)
        v_des_L = np.stack([d["v_des_L"] for d in data], axis=0)
        v_des_R = np.stack([d["v_des_R"] for d in data], axis=0)

        fig2, axes2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig2.suptitle(f"{title_prefix} – Base-frame Metrics")

        # Row 1: TCP positions vs grasp positions (x,y,z)
        labels_xyz = ["x", "y", "z"]
        for j in range(2):  # 0 = L, 1 = R
            ax = axes2[0, j]
            if j == 0:
                p = p_L
                g = grasp_L
                side = "LEFT"
            else:
                p = p_R
                g = grasp_R
                side = "RIGHT"

            for k in range(3):
                ax.plot(t, p[:, k], label=f"p_{labels_xyz[k]}")
                ax.plot(t, g[:, k], "--", label=f"grasp_{labels_xyz[k]}" if k == 0 else None)

            ax.set_ylabel(f"{side} TCP / grasp [m]")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(ncol=3, fontsize=8)

        # Row 2: distance between TCP and grasp (d_L, d_R)
        ax = axes2[1, 0]
        ax.plot(t, d_L, label="d_L (along -n)")
        ax.set_ylabel("d_L [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes2[1, 1]
        ax.plot(t, d_R, label="d_R (along -n)")
        ax.set_ylabel("d_R [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Row 3: commanded vs measured TCP linear velocity in base frame (norm)
        v_tcp_L_norm = np.linalg.norm(v_tcp_L[:, :3], axis=1)
        v_tcp_R_norm = np.linalg.norm(v_tcp_R[:, :3], axis=1)
        v_des_L_norm = np.linalg.norm(v_des_L[:, :3], axis=1)
        v_des_R_norm = np.linalg.norm(v_des_R[:, :3], axis=1)

        ax = axes2[2, 0]
        ax.plot(t, v_tcp_L_norm, label="|v_tcp_L| meas")
        ax.plot(t, v_des_L_norm, "--", label="|v_des_L| cmd")
        ax.set_ylabel("LEFT |v| [m/s]")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes2[2, 1]
        ax.plot(t, v_tcp_R_norm, label="|v_tcp_R| meas")
        ax.plot(t, v_des_R_norm, "--", label="|v_des_R| cmd")
        ax.set_ylabel("RIGHT |v| [m/s]")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

        # --- saving ---
        os.makedirs("plots", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        f1 = os.path.join("plots", f"{title_prefix}_normal_{ts}.png")
        f2 = os.path.join("plots", f"{title_prefix}_base_{ts}.png")

        fig1.savefig(f1, dpi=200)
        fig2.savefig(f2, dpi=200)

        print(f"[plot_force_profile] Saved:\n  {f1}\n  {f2}")



if __name__ == "__main__":
    # Stiffness for impedance tracking (unchanged)
    K = np.diag([10, 10, 10, 0.2, 0.2, 0.2])
    
    robot_L = URImpedanceController("192.168.1.33", K=K)
    robot_R = URImpedanceController("192.168.1.66", K=K)

    # Weights
    W_imp = diag6([1.0, 1.0, 1.0, 1e3, 1e3, 1e3])
    #W_imp = diag6([0,0,0,0,0,0])
    W_grasp = diag6([5e1, 5e1, 5e1, 1e3, 1e3, 1e3])
    #W_grasp = diag6([0, 0, 0, 0, 0, 0])
    lambda_reg = 1e-6

    M_a = 27.0
    K_a = 300.0
    D_a = 2400.0  # or 2*sqrt(M_a*K_a)
    v_max = 0.05
    Fn_ref = 20.0

    traj_path = "motion_planner/trajectories_old/lift_100.npz"
    Hz = 50

    ctrl = DualArmImpedanceAdmittanceQP(
        robot_L=robot_L,
        robot_R=robot_R,
        Hz=Hz,
        trajectory_npz_path=traj_path,
        W_imp_L=W_imp, W_imp_R=W_imp,
        W_grasp_L=W_grasp, W_grasp_R=W_grasp,
        lambda_reg=lambda_reg,
        M_a=M_a, D_a=D_a, K_a=K_a,
        v_max=v_max, ref_force=Fn_ref
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

        # run controller
        ctrl.run()

        ctrl.plot_taskspace("L")
        ctrl.plot_taskspace("R")
        ctrl.plot_jointspace("L")
        ctrl.plot_jointspace("R")
        ctrl.plot_qp_objective()
        ctrl.plot_qp_performance()
        ctrl.plot_force_profile()

    except KeyboardInterrupt:
        print("Interrupted")
        robot_R.stop_control()
        robot_L.stop_control()
    finally:
        try:
            robot_L.disconnect(); robot_R.disconnect()
        except: pass
        print("Robots disconnected.")
