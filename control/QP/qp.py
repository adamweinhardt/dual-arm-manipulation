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

from control.PID.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity, _as_rowvec_1d


# =============================================================
# Helpers
# =============================================================

def diag6(vals):
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if vals.size == 1:
        return np.eye(6) * vals.item()
    if vals.size == 3:
        return np.diag([vals[0], vals[1], vals[2], vals[0], vals[1], vals[2]])
    if vals.size == 6:
        return np.diag(vals)
    raise ValueError("diag6 expects 1, 3, or 6 values")


# =============================================================
# Robot wrapper (same API as before)
# =============================================================
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

    # def get_Lambda(self, J, M, lam2_base=1e-5, cond_ref=5):
    #     # damped operational-space "inverse inertia"
    #     Minv = np.linalg.inv(M)
    #     A = J @ Minv @ J.T
    #     A = 0.5 * (A + A.T)
    #     w, V = np.linalg.eigh(A)
    #     w = np.clip(w, 1e-6, 1e6)
    #     wmin = float(np.min(w)) if w.size else 0.0
    #     wmax = float(np.max(w)) if w.size else 0.0
    #     condA = (wmax / max(wmin, 1e-12)) if wmax > 0 else 1.0
    #     lam2 = lam2_base * max(1.0, condA / cond_ref)
    #     winv_damped = w / (w * w + lam2)
    #     Lam = (V * winv_damped) @ V.T
    #     return 0.5 * (Lam + Lam.T)

    def get_Lambda(self, J, M,  regularization: float = 1e-10):
        try:
            # Check shapes
            if M.shape[0] != M.shape[1]:
                raise ValueError("M_i must be a square matrix.")
            if J.shape[1] != M.shape[0]:
                raise ValueError("J_i and M_i dimensions do not align.")

            # Compute M_i inverse safely
            M_inv = np.linalg.pinv(M)  # pseudoinverse is safer than inv

            # Compute the intermediate matrix
            inner = J @ M_inv @ J.T

            # Add a small regularization term to ensure invertibility
            inner_reg = inner + regularization * np.eye(inner.shape[0])

            # Compute the final inverse
            Lambda_i = np.linalg.pinv(inner_reg)
            return Lambda_i

        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Matrix inversion failed: {e}")


    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda)
        S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        return np.real_if_close(0.5 * (D + D.T))

    def wrench_desired(self, K, D, e_p, e_r, e_v, e_w):
        X_err = np.hstack((e_p, e_r))
        Xd_err = np.hstack((e_v, e_w))
        return D @ Xd_err + K @ X_err


# =============================================================
# Combined Dual-Arm QP: Impedance + Admittance (single optimization)
# + Trajectory tracking after 3 s (computed inside, no extra class)
# =============================================================
class DualArmImpedanceAdmittanceQP:
    """
    Objective (per arm in one QP):
        min  Σ_i  || W_imp   (J_i qddot_i + Jdot_i qdot_i - a_i_des) ||^2
                 +|| W_grasp (J_i (qdot_i + qddot_i dt) - xdot_i*)   ||^2
                 + λ ||qddot_i||^2
        s.t. one-step joint pos/vel/acc bounds.
    """

    def __init__(self, robotL: URImpedanceController, robotR: URImpedanceController, Hz: int,
                 W_imp=None, W_grasp=None, lam_reg: float = 1e-8,
                 admittance_gain: float = 3e-4, v_max: float = 0.05,
                 F_n_star: float = 20.0,
                 trajectory_path: str | None = None,
                 ref_start_delay_s: float = 3.0):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz

        self.W_imp_np = np.asarray(W_imp, dtype=float)
        self.W_grasp_np = np.asarray(W_grasp, dtype=float)
        self.lam_reg = float(lam_reg)

        # admittance params
        self.k_f = float(admittance_gain)
        self.v_max = float(v_max)
        self.F_n_star = float(F_n_star)

        # reference tracking storage (from your npz format)
        self.trajectory_npz = None
        self.traj_len = 0
        self.ref_start_delay_s = float(ref_start_delay_s)
        self.position_ref_L = None; self.velocity_ref_L = None
        self.rotation_ref_L = None; self.angular_velocity_ref_L = None
        self.position_ref_R = None; self.velocity_ref_R = None
        self.rotation_ref_R = None; self.angular_velocity_ref_R = None
        self.trajectory_path = trajectory_path

        # symmetric box limits (replace with per-joint vectors if you have them)
        self.joint_pose_limit  = np.deg2rad(360.0)
        self.joint_speed_limit = np.deg2rad(180.0)
        self.joint_accel_limit = np.deg2rad(500.0)

        # logging / perf
        self.control_data = []
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

    # ----------------------------- Trajectory init (your format) -----------------------------
    def load_and_init_refs(self, npz_path: str):
        """
        Use trajectory deltas in WORLD frame. For each arm:
        position_ref[t]         = p_gripper_init + p_box[t]
        velocity_ref[t]         = v_box[t]
        rotation_ref[t]         = R_gripper_init @ R_box[t]   # lock orientation
        angular_velocity_ref[t] = w_box[t]
        """
        self.trajectory_npz = np.load(npz_path)
        p_box = self.trajectory_npz['position']            # (T,3)
        v_box = self.trajectory_npz['linear_velocity']     # (T,3)
        R_box = self.trajectory_npz['rotation_matrices']   # (T,3,3)
        w_box = self.trajectory_npz['angular_velocity']    # (T,3)
        self.traj_len = int(p_box.shape[0])

        # read initial world pose/orientation of both TCPs
        stateL = self.robotL.get_state()
        pL_init = np.array(stateL["gripper_world"][:3])
        RL_init = RR.from_rotvec(stateL["pose"][3:6]).as_matrix()

        stateR = self.robotR.get_state()
        pR_init = np.array(stateR["gripper_world"][:3])
        RR_init = RR.from_rotvec(stateR["pose"][3:6]).as_matrix()

        # allocate
        self.position_ref_L         = np.zeros_like(p_box)
        self.velocity_ref_L         = np.zeros_like(v_box)
        self.rotation_ref_L         = np.zeros((self.traj_len, 3, 3))
        self.angular_velocity_ref_L = np.zeros_like(w_box)

        self.position_ref_R         = np.zeros_like(p_box)
        self.velocity_ref_R         = np.zeros_like(v_box)
        self.rotation_ref_R         = np.zeros((self.traj_len, 3, 3))
        self.angular_velocity_ref_R = np.zeros_like(w_box)

        for t in range(self.traj_len):
            # LEFT
            self.position_ref_L[t]         = pL_init + p_box[t]
            self.velocity_ref_L[t]         = v_box[t]
            self.rotation_ref_L[t]         = RL_init @ R_box[t]    # lock orientation (your choice)
            self.angular_velocity_ref_L[t] = w_box[t]
            # RIGHT
            self.position_ref_R[t]         = pR_init + p_box[t]
            self.velocity_ref_R[t]         = v_box[t]
            self.rotation_ref_R[t]         = RR_init @ R_box[t]
            self.angular_velocity_ref_R[t] = w_box[t]

    # ---------------------------------------------------------
    # QP build
    # ---------------------------------------------------------
    def build_qp(self, dt):
        """
        QP build including optional posture task regularization.
        Keeps the same impedance + grasping formulation, but adds a weak
        joint-space "posture" term to stabilize the configuration and prevent drift.
        """
        n = 6
        self.qddot_L = cp.Variable(n, name="qddot_L")
        self.qddot_R = cp.Variable(n, name="qddot_R")

        # ===============================================================
        # Parameters (updated each loop)
        # ===============================================================
        self.J_L_p = cp.Parameter((6, n), name="J_L")
        self.Jdot_L_p = cp.Parameter((6, n), name="Jdot_L")
        self.qdot_L_p = cp.Parameter(n, name="qdot_L")
        self.xddot_des_L_p = cp.Parameter(6, name="a_des_L")
        self.xdot_des_L_p = cp.Parameter(6, name="xdot_star_L")
        self.q_L_p = cp.Parameter(n, name="q_L")

        self.J_R_p = cp.Parameter((6, n), name="J_R")
        self.Jdot_R_p = cp.Parameter((6, n), name="Jdot_R")
        self.qdot_R_p = cp.Parameter(n, name="qdot_R")
        self.xddot_des_R_p = cp.Parameter(6, name="a_des_R")
        self.xdot_des_R_p = cp.Parameter(6, name="xdot_star_R")
        self.q_R_p = cp.Parameter(n, name="q_R")

        self.W_imp_c = cp.Parameter((6, 6), name="W_imp")
        self.W_grasp_c = cp.Parameter((6, 6), name="W_grasp")

        self.dt_c = cp.Constant(float(dt))
        self.dt2_c = cp.Constant(float(0.5 * dt * dt))

        # ===============================================================
        # Posture task (new)
        # ===============================================================
        self.q_post_L_p    = cp.Parameter(n, name="q_post_L")
        self.qdot_post_L_p = cp.Parameter(n, name="qdot_post_L")
        self.q_post_R_p    = cp.Parameter(n, name="q_post_R")
        self.qdot_post_R_p = cp.Parameter(n, name="qdot_post_R")

        self.S_post_L = cp.Parameter((n, n), name="S_post_L")
        self.S_post_R = cp.Parameter((n, n), name="S_post_R")

        self.k_pos_c  = cp.Constant(15.0)    # posture "stiffness" gain
        self.w_post_c = cp.Constant(2e4)   # posture cost weight (small)

        # ===============================================================
        # State update expressions
        # ===============================================================
        qdot_next_L = self.qdot_L_p + self.qddot_L * self.dt_c
        qdot_next_R = self.qdot_R_p + self.qddot_R * self.dt_c
        q_next_L = self.q_L_p + self.qdot_L_p * self.dt_c + self.qddot_L * self.dt2_c
        q_next_R = self.q_R_p + self.qdot_R_p * self.dt_c + self.qddot_R * self.dt2_c

        # ===============================================================
        # Task-space impedance and grasping errors
        # ===============================================================
        e_imp_L = self.J_L_p @ self.qddot_L + self.Jdot_L_p @ self.qdot_L_p - self.xddot_des_L_p
        e_imp_R = self.J_R_p @ self.qddot_R + self.Jdot_R_p @ self.qdot_R_p - self.xddot_des_R_p

        xdot_next_L = self.J_L_p @ qdot_next_L
        xdot_next_R = self.J_R_p @ qdot_next_R
        e_grasp_L = xdot_next_L - self.xdot_des_L_p
        e_grasp_R = xdot_next_R - self.xdot_des_R_p

        # ===============================================================
        # Posture task errors
        # ===============================================================
        # β = 2√k (q̇*_post − q̇) + k (q*_post − q)
        beta_L = 2.0 * cp.sqrt(self.k_pos_c) * (self.qdot_post_L_p - self.qdot_L_p) \
                + self.k_pos_c * (self.q_post_L_p - self.q_L_p)
        beta_R = 2.0 * cp.sqrt(self.k_pos_c) * (self.qdot_post_R_p - self.qdot_R_p) \
                + self.k_pos_c * (self.q_post_R_p - self.q_R_p)

        e_post_L = self.S_post_L @ self.qddot_L - beta_L
        e_post_R = self.S_post_R @ self.qddot_R - beta_R

        # ===============================================================
        # Objective
        # ===============================================================
        obj_L = (
            cp.sum_squares(self.W_imp_c @ e_imp_L)
            + cp.sum_squares(self.W_grasp_c @ e_grasp_L)
            + self.lam_reg * cp.sum_squares(self.qddot_L)
            + self.w_post_c * cp.sum_squares(e_post_L)     # posture regularization
        )

        obj_R = (
            cp.sum_squares(self.W_imp_c @ e_imp_R)
            + cp.sum_squares(self.W_grasp_c @ e_grasp_R)
            + self.lam_reg * cp.sum_squares(self.qddot_R)
            + self.w_post_c * cp.sum_squares(e_post_R)
        )

        obj = obj_L + obj_R

        # ===============================================================
        # Constraints
        # ===============================================================
        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_acc_lim = self.joint_accel_limit

        cons = [
            -q_pos_lim <= q_next_L, q_next_L <= q_pos_lim,
            -q_pos_lim <= q_next_R, q_next_R <= q_pos_lim,
            -q_vel_lim <= qdot_next_L, qdot_next_L <= q_vel_lim,
            -q_vel_lim <= qdot_next_R, qdot_next_R <= q_vel_lim,
            -q_acc_lim <= self.qddot_L, self.qddot_L <= q_acc_lim,
            -q_acc_lim <= self.qddot_R, self.qddot_R <= q_acc_lim,
        ]

        # ===============================================================
        # Build QP
        # ===============================================================
        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(
            eps_abs=6e-6,
            eps_rel=6e-6,
            alpha=1.6,
            max_iter=20000,
            adaptive_rho=True,
            adaptive_rho_interval=45,
            polish=True,
            check_termination=10,
            warm_start=True,
        )


    # ---------------------------------------------------------
    # Main control loop
    # ---------------------------------------------------------
    def run(self, timeout_s: float | None = None):
        """
        If timeout_s is given, stop after that many seconds and auto-plot (if enabled).
        First ref_start_delay_s seconds: pure grasping (admittance only).
        After that: use trajectory references initialized from current TCPs.
        """
        self.load_and_init_refs(self.trajectory_path)
        trajectory_started = False
        t_idx = 0
        time.sleep(0.1)

        dt = 1.0 / self.Hz
        self.build_qp(dt)
        self.control_stop = threading.Event()
        i = 0

        # constant W matrices
        self.W_imp_c.value = self.W_imp_np
        self.W_grasp_c.value = self.W_grasp_np

        self._ctrl_start_wall = time.perf_counter()
        self._last_log_wall = self._ctrl_start_wall

        # prepare integrated q_cmd traces for plotting
        self._qcmd_L = self.robotL.get_q().copy()
        self._qcmd_R = self.robotR.get_q().copy()

        # --- posture reference setup ---
        qL0 = self.robotL.get_q().copy()
        qR0 = self.robotR.get_q().copy()

        self.q_post_L = self.robotL.home_joints
        self.qdot_post_L = np.zeros_like(qL0)
        self.q_post_R = self.robotR.home_joints
        self.qdot_post_R = np.zeros_like(qR0)

        # select which joints to stabilize (1=on, 0=off)
        S_mask = np.diag([0, 1, 1, 1, 1, 0]).astype(float)
        self.S_post_L.value = S_mask
        self.S_post_R.value = S_mask

        self.q_post_L_p.value = self.q_post_L
        self.qdot_post_L_p.value = self.qdot_post_L
        self.q_post_R_p.value = self.q_post_R
        self.qdot_post_R_p.value = self.qdot_post_R

        try:
            while not self.control_stop.is_set():
                loop_t0 = time.perf_counter()
                elapsed_total = loop_t0 - self._ctrl_start_wall

                # timeout
                if timeout_s is not None and elapsed_total >= timeout_s:
                    print(f"[COMBINED QP] timeout reached ({timeout_s:.2f}s) → stopping.")
                    break

                # ---------- LEFT state ----------
                q_L = self.robotL.get_q(); qdL = self.robotL.get_qdot()
                state_L = self.robotL.get_state()
                R_L = RR.from_rotvec(state_L["pose"][3:6]).as_matrix()
                n_L = -R_L[:, 1]
                Fw_L = np.array(state_L.get("force_world", [0, 0, 0, 0, 0, 0])[:3])
                F_n_L = float(n_L @ Fw_L)
                J_L = self.robotL.get_J(q_L)
                Jd_L = self.robotL.get_Jdot(q_L, qdL)
                M_L = self.robotL.get_M(q_L)
                Lam_L = self.robotL.get_Lambda(J_L, M_L)
                D_L = self.robotL.get_D(self.robotL.K, Lam_L)

                p_L = np.array(state_L["gripper_world"][:3])
                v_L = np.array(state_L["speed_world"][:3])
                w_L = np.array(state_L["speed_world"][3:6])
                R_cur_L = R_L

                # ---------- RIGHT state ----------
                q_R = self.robotR.get_q(); qdR = self.robotR.get_qdot()
                state_R = self.robotR.get_state()
                R_R = RR.from_rotvec(state_R["pose"][3:6]).as_matrix()
                n_R = -R_R[:, 1]
                Fw_R = np.array(state_R.get("force_world", [0, 0, 0, 0, 0, 0])[:3])
                F_n_R = float(n_R @ Fw_R)
                J_R = self.robotR.get_J(q_R)
                Jd_R = self.robotR.get_Jdot(q_R, qdR)
                M_R = self.robotR.get_M(q_R)
                Lam_R = self.robotR.get_Lambda(J_R, M_R)
                D_R = self.robotR.get_D(self.robotR.K, Lam_R)

                p_R = np.array(state_R["gripper_world"][:3])
                v_R = np.array(state_R["speed_world"][:3])
                w_R = np.array(state_R["speed_world"][3:6])
                R_cur_R = R_R

                # ---------- References (your format, after delay) ----------
                if (self.traj_len > 0) and (elapsed_total >= self.ref_start_delay_s):
                    trajectory_started = True
                    t_idx = int(np.clip((elapsed_total - self.ref_start_delay_s) * self.Hz, 0, self.traj_len - 1))

                    p_ref_L = self.position_ref_L[t_idx]
                    v_ref_L = self.velocity_ref_L[t_idx]
                    R_ref_L = self.rotation_ref_L[t_idx]
                    w_ref_L = self.angular_velocity_ref_L[t_idx]

                    p_ref_R = self.position_ref_R[t_idx]
                    v_ref_R = self.velocity_ref_R[t_idx]
                    R_ref_R = self.rotation_ref_R[t_idx]
                    w_ref_R = self.angular_velocity_ref_R[t_idx]
                else:
                    # hold-current during grasping-only phase (zero error)
                    p_ref_L, v_ref_L, w_ref_L, R_ref_L = p_L, v_L, w_L, R_cur_L
                    p_ref_R, v_ref_R, w_ref_R, R_ref_R = p_R, v_R, w_R, R_cur_R

                # ---------- Impedance desired spatial acceleration ----------
                e_p_L = p_ref_L - p_L
                e_v_L = v_ref_L - v_L
                e_R_L = R_ref_L @ R_cur_L.T
                e_r_L = pin.log3(e_R_L)
                e_w_L = w_ref_L - w_L
                f_des_L = self.robotL.wrench_desired(self.robotL.K, D_L, e_p_L, e_r_L, e_v_L, e_w_L)
                a_des_L = np.linalg.solve(Lam_L, f_des_L)

                e_p_R = p_ref_R - p_R
                e_v_R = v_ref_R - v_R
                e_R_R = R_ref_R @ R_cur_R.T
                e_r_R = pin.log3(e_R_R)
                e_w_R = w_ref_R - w_R
                f_des_R = self.robotR.wrench_desired(self.robotR.K, D_R, e_p_R, e_r_R, e_v_R, e_w_R)
                a_des_R = np.linalg.solve(Lam_R, f_des_R)

                # ---------- Admittance (normal-direction only) ----------
                v_n_L_star = np.clip(self.k_f * (self.F_n_star - F_n_L), -self.v_max, self.v_max)
                v_n_R_star = np.clip(self.k_f * (self.F_n_star - F_n_R), -self.v_max, self.v_max)
                xdot_star_L = np.hstack([v_n_L_star * n_L, np.zeros(3)])
                xdot_star_R = np.hstack([v_n_R_star * n_R, np.zeros(3)])

                # ---------- Feed QP ----------
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.Jdot_L_p.value = _freeze_sparsity(Jd_L)
                self.qdot_L_p.value = qdL
                self.xddot_des_L_p.value = a_des_L
                self.xdot_des_L_p.value = xdot_star_L
                self.q_L_p.value = q_L

                self.J_R_p.value = _freeze_sparsity(J_R)
                self.Jdot_R_p.value = _freeze_sparsity(Jd_R)
                self.qdot_R_p.value = qdR
                self.xddot_des_R_p.value = a_des_R
                self.xdot_des_R_p.value = xdot_star_R
                self.q_R_p.value = q_R

                # ---------- Solve ----------
                _t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                _solve_dt = time.perf_counter() - _t0
                self._win_solver_time += _solve_dt
                self._total_solver_time += _solve_dt

                ok = (self.qp.status in ("optimal", "optimal_inaccurate"))
                if ok:
                    qddL = _as_rowvec_1d(self.qddot_L.value, "qddot_L", length=6)
                    qddR = _as_rowvec_1d(self.qddot_R.value, "qddot_R", length=6)
                else:
                    qddL = np.zeros(6); qddR = np.zeros(6)

                qdot_cmd_L = qdL + qddL * dt
                qdot_cmd_R = qdR + qddR * dt

                # ---------- Send ----------
                self.robotL.speedJ(qdot_cmd_L.tolist(), dt)
                self.robotR.speedJ(qdot_cmd_R.tolist(), dt)

                # ---------- Update integrated positions for plotting ----------
                self._qcmd_L = self._qcmd_L + qdot_cmd_L * dt
                self._qcmd_R = self._qcmd_R + qdot_cmd_R * dt

                # ---------- Objective breakdown (values) ----------
                e_imp_L_val = J_L @ qddL + Jd_L @ qdL - a_des_L
                e_imp_R_val = J_R @ qddR + Jd_R @ qdR - a_des_R
                e_grasp_L_val = J_L @ (qdL + qddL * dt) - xdot_star_L
                e_grasp_R_val = J_R @ (qdR + qddR * dt) - xdot_star_R

                W_imp = self.W_imp_c.value; W_grasp = self.W_grasp_c.value
                imp_L_term = float((W_imp @ e_imp_L_val).T @ (W_imp @ e_imp_L_val))
                imp_R_term = float((W_imp @ e_imp_R_val).T @ (W_imp @ e_imp_R_val))
                grasp_L_term = float((W_grasp @ e_grasp_L_val).T @ (W_grasp @ e_grasp_L_val))
                grasp_R_term = float((W_grasp @ e_grasp_R_val).T @ (W_grasp @ e_grasp_R_val))
                reg_term = float(self.lam_reg) * (float(qddL @ qddL) + float(qddR @ qddR))
                obj_total = imp_L_term + imp_R_term + grasp_L_term + grasp_R_term + reg_term

                # ---------- Per-arm tcp dicts for plotting ----------
                rvec_L      = RR.from_matrix(R_cur_L).as_rotvec()
                rvec_ref_L  = RR.from_matrix(R_ref_L).as_rotvec()

                if i % 20 == 0 and trajectory_started:
                    # ---------- compute helpful debug values ----------
                    condJL = np.linalg.cond(J_L)
                    condJR = np.linalg.cond(J_R)

                    e_pL_mag = np.linalg.norm(e_p_L)
                    e_pR_mag = np.linalg.norm(e_p_R)
                    e_rL_mag = np.linalg.norm(e_r_L)
                    e_rR_mag = np.linalg.norm(e_r_R)
                    e_vL_mag = np.linalg.norm(e_v_L)
                    e_vR_mag = np.linalg.norm(e_v_R)
                    e_wL_mag = np.linalg.norm(e_w_L)
                    e_wR_mag = np.linalg.norm(e_w_R)

                    aL_mag = np.linalg.norm(a_des_L)
                    aR_mag = np.linalg.norm(a_des_R)
                    qddL_mag = np.linalg.norm(qddL)
                    qddR_mag = np.linalg.norm(qddR)

                    # twist consistency (jacobian vs measured)
                    twist_pred_L = J_L @ qdL
                    twist_pred_R = J_R @ qdR
                    twist_meas_L = np.hstack([v_L, w_L])
                    twist_meas_R = np.hstack([v_R, w_R])
                    twist_err_L = twist_pred_L - twist_meas_L
                    twist_err_R = twist_pred_R - twist_meas_R
                    twist_errL_lin = np.linalg.norm(twist_err_L[:3])
                    twist_errR_lin = np.linalg.norm(twist_err_R[:3])
                    twist_errL_ang = np.linalg.norm(twist_err_L[3:])
                    twist_errR_ang = np.linalg.norm(twist_err_R[3:])

                    # Λ spectrum
                    eigL = np.linalg.eigvalsh(Lam_L)
                    eigR = np.linalg.eigvalsh(Lam_R)
                    lamL_min, lamL_max = eigL.min(), eigL.max()
                    lamR_min, lamR_max = eigR.min(), eigR.max()

                    # qp objective parts
                    obj_parts = f"impL={imp_L_term:.2e}, impR={imp_R_term:.2e}, reg={reg_term:.2e}, total={obj_total:.2e}"

                    # ramp factor
                    print(f"\n[DBG] t={elapsed_total:6.2f}s ")
                    print(f"  cond(J) [L,R]=({condJL:6.2f}, {condJR:6.2f}) | Λ[min,max]_L=({lamL_min:.2e},{lamL_max:.2e})  "
                        f"Λ[min,max]_R=({lamR_min:.2e},{lamR_max:.2e})")
                    print(f"  e_p [L,R]=({e_pL_mag:.4f},{e_pR_mag:.4f})  e_r [L,R]=({e_rL_mag:.4f},{e_rR_mag:.4f})")
                    print(f"  e_v [L,R]=({e_vL_mag:.4f},{e_vR_mag:.4f})  e_w [L,R]=({e_wL_mag:.4f},{e_wR_mag:.4f})")
                    print(f"  ‖a_des‖ [L,R]=({aL_mag:.2e},{aR_mag:.2e})  ‖q̈‖ [L,R]=({qddL_mag:.2e},{qddR_mag:.2e})")
                    print(f"  twist_err_lin [L,R]=({twist_errL_lin:.2e},{twist_errR_lin:.2e})  "
                        f"twist_err_ang [L,R]=({twist_errL_ang:.2e},{twist_errR_ang:.2e})")
                    print(f"  {obj_parts}")
                    print(f"  status={self.qp.status} | solver_time={_solve_dt*1e3:.2f} ms | Hz≈{1.0/max(elapsed,1e-9):5.2f}\n")


                tcp_L = {
                    "p":        p_L, "v": v_L, "w": w_L, "rvec": rvec_L,
                    "p_ref":    p_ref_L, "v_ref": v_ref_L, "w_ref": w_ref_L, "rvec_ref": rvec_ref_L,
                    "e_p":      p_ref_L - p_L,
                    "e_v":      v_ref_L - v_L,
                    "e_w":      w_ref_L - w_L,
                    "e_r":      pin.log3(R_ref_L @ R_cur_L.T),
                }

                rvec_R      = RR.from_matrix(R_cur_R).as_rotvec()
                rvec_ref_R  = RR.from_matrix(R_ref_R).as_rotvec()
                tcp_R = {
                    "p":        p_R, "v": v_R, "w": w_R, "rvec": rvec_R,
                    "p_ref":    p_ref_R, "v_ref": v_ref_R, "w_ref": w_ref_R, "rvec_ref": rvec_ref_R,
                    "e_p":      p_ref_R - p_R,
                    "e_v":      v_ref_R - v_R,
                    "e_w":      w_ref_R - w_R,
                    "e_r":      pin.log3(R_ref_R @ R_cur_R.T),
                }

                # ---------- Log ----------
                self.control_data.append({
                    "t": time.time(), "i": i, "status": self.qp.status,
                    "obj": obj_total,
                    # forces/admittance
                    "F_n_L": F_n_L, "F_n_R": F_n_R, "F_n_star": self.F_n_star,
                    "v_n_L_star": float(v_n_L_star), "v_n_R_star": float(v_n_R_star),
                    # objective terms
                    "obj_break": {
                        "imp_L": imp_L_term, "imp_R": imp_R_term,
                        "grasp_L": grasp_L_term, "grasp_R": grasp_R_term,
                        "reg": reg_term, "total": obj_total,
                    },
                    # per-arm tcp blobs for plotting
                    "tcp_L": tcp_L,
                    "tcp_R": tcp_R,
                    # joints and commands (meas + qp + integrated)
                    "q_L": q_L, "q_R": q_R,
                    "qdot_L": qdL, "qdot_R": qdR,
                    "qddot_L": qddL, "qddot_R": qddR,
                    "qdot_cmd_L": qdot_cmd_L, "qdot_cmd_R": qdot_cmd_R,
                    "q_cmd_L": self._qcmd_L.copy(), "q_cmd_R": self._qcmd_R.copy(),
                })

                # ---------- perf / pacing ----------
                elapsed = time.perf_counter() - loop_t0
                self._win_loop_time += elapsed; self._win_iters += 1; self._total_iters += 1
                if elapsed > dt + 1e-4:
                    self._win_deadline_miss += 1; self._total_deadline_miss += 1

                now = time.perf_counter()
                if now - self._last_log_wall >= self.log_every_s and self._win_iters > 0:
                    avg_period = self._win_loop_time / self._win_iters
                    avg_hz = (1.0 / avg_period) if avg_period > 0 else float('nan')
                    avg_solver_ms = (self._win_solver_time / self._win_iters) * 1000.0
                    miss_pct = (100.0 * self._win_deadline_miss / self._win_iters)
                    print(f"[COMBINED QP] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms | miss {miss_pct:4.1f}% | "
                          f"F_L={F_n_L:6.2f} F_R={F_n_R:6.2f} | obj={obj_total:.3e}")
                    self._win_iters = 0; self._win_loop_time = 0.0; self._win_solver_time = 0.0
                    self._win_deadline_miss = 0; self._last_log_wall = now

                if (t_idx >= self.traj_len - 1) and (self.traj_len > 0):
                    self.robotL.speedStop(); self.robotR.speedStop()
                    print(f"[COMBINED QP] reached end of trajectory at t={elapsed_total:.2f}s → stopping.")
                    break               

                time.sleep(max(0, dt - elapsed))
                i += 1

        finally:
            # graceful stop
            try:
                self.robotL.speedStop(); self.robotR.speedStop()
            except Exception:
                pass
            total_time = time.perf_counter() - self._ctrl_start_wall
            print(f"[COMBINED QP SUMMARY] Ran {self._total_iters} iters @ {self._total_iters/max(total_time,1e-9):.1f} Hz")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    def _plot_taskspace_for_key(self, key_tcp: str, robot_tag: str, title_prefix="TaskspaceTracking_SingleArm"):
        if not self.control_data:
            print("No control_data to plot."); return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t  = np.array(ts) - ts[0]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X","Y","Z"]

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
        fig.suptitle(f"{title_prefix} – {robot_tag} – {timestamp_str}", fontsize=14, y=0.99)

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
        fname = f"plots/{title_prefix.lower()}_{robot_tag.replace(' ','_')}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_taskspace(self, title_prefix="TaskspaceTracking_SingleArm"):
        self._plot_taskspace_for_key("tcp_L", "Robot L", title_prefix)
        self._plot_taskspace_for_key("tcp_R", "Robot R", title_prefix)

    def _plot_jointspace_for_arm(self, arm: str, title_prefix="QP_and_Jointspace_SingleArm"):
        if not self.control_data:
            print("No control_data to plot."); return

        os.makedirs("plots", exist_ok=True)
        ts = [d["t"] for d in self.control_data]
        t  = np.array(ts) - ts[0]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        suffix = "_L" if arm.upper()=="L" else "_R"
        Q        = np.vstack([d[f"q{suffix}"]         for d in self.control_data])
        QDOT     = np.vstack([d[f"qdot{suffix}"]      for d in self.control_data])
        QDDOT    = np.vstack([d[f"qddot{suffix}"]     for d in self.control_data])
        QDOT_cmd = np.vstack([d[f"qdot_cmd{suffix}"]  for d in self.control_data])
        Q_cmd    = np.vstack([d[f"q_cmd{suffix}"]     for d in self.control_data])

        n = Q.shape[1]; jlabels = [f"J{i+1}" for i in range(n)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        fig.suptitle(f"{title_prefix} – Robot {arm.upper()} – {timestamp_str}", fontsize=14, y=0.98)

        ax = axes[0, 0]
        for j in range(n):
            ax.plot(t, Q[:, j], label=f"{jlabels[j]} meas")
            ax.plot(t, Q_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd(int)")
        ax.set_title("Joint Positions: measured vs integrated-from-QP")
        ax.set_ylabel("rad"); ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(2*n, 6), fontsize=8)

        ax = axes[0, 1]
        for j in range(n):
            ax.plot(t, QDOT[:, j], label=f"{jlabels[j]} meas")
            ax.plot(t, QDOT_cmd[:, j], "--", alpha=0.9, label=f"{jlabels[j]} cmd")
        ax.set_title("Joint Velocities: measured vs commanded")
        ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(2*n, 6), fontsize=8)

        ax = axes[1, 0]
        for j in range(n):
            ax.plot(t, QDDOT[:, j], label=jlabels[j])
        ax.set_title("Joint Accelerations qddot (QP)")
        ax.set_xlabel("time [s]"); ax.set_ylabel("rad/s²"); ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(n, 6), fontsize=8)

        # QP status class
        status = [d.get("status", "") for d in self.control_data]
        def classify_status(s):
            s = (s or "").lower()
            if s == "optimal": return 0
            if s in ("optimal_inaccurate","user_limit","max_iters_reached",
                     "iteration_limit_reached","user_limit_reached"): return 1
            if any(k in s for k in ["infeasible","unbounded","solver_error",
                                    "error","dual_infeasible","primal_infeasible"]): return 2
            return 2
        class_vals = np.array([classify_status(s) for s in status], dtype=float)

        ax = axes[1, 1]
        ax.step(t, class_vals, where="post")
        ax.set_title("QP solve class (0=good,1=warn,2=bad)")
        ax.set_xlabel("time [s]"); ax.set_yticks([0,1,2]); ax.set_yticklabels(["good","warn","bad"])
        ax.set_ylim(-0.5, 2.5); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_{arm.upper()}_{timestamp_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_jointspace(self, title_prefix="QP_and_Jointspace_SingleArm"):
        self._plot_jointspace_for_arm("L", title_prefix)
        self._plot_jointspace_for_arm("R", title_prefix)

    def plot_force_tracking(self, title_prefix="ForceTracking_DualArm"):
        if not self.control_data:
            print("No control_data to plot."); return
        os.makedirs("plots", exist_ok=True)

        ts = np.array([d["t"] for d in self.control_data])
        t  = ts - ts[0]
        F_ref = np.array([d["F_n_star"]  for d in self.control_data])
        F_L   = np.array([d["F_n_L"]     for d in self.control_data])
        F_R   = np.array([d["F_n_R"]     for d in self.control_data])
        vL    = np.array([d["v_n_L_star"] for d in self.control_data])
        vR    = np.array([d["v_n_R_star"] for d in self.control_data])

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.suptitle(f"{title_prefix} – {ts_str}", fontsize=14, y=0.98)

        ax = axes[0]
        ax.plot(t, F_L, label="F_n_L")
        ax.plot(t, F_ref, "--", label="F*_n")
        ax2 = ax.twinx()
        ax2.plot(t, vL, ":", label="v*_n L")
        ax.set_title("Left arm"); ax.set_ylabel("Force [N]"); ax.grid(True, alpha=0.3)
        ax2.set_ylabel("v*_n [m/s]")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")

        ax = axes[1]
        ax.plot(t, F_R, label="F_n_R")
        ax.plot(t, F_ref, "--", label="F*_n")
        ax2 = ax.twinx()
        ax2.plot(t, vR, ":", label="v*_n R")
        ax.set_title("Right arm"); ax.set_xlabel("time [s]"); ax.set_ylabel("Force [N]"); ax.grid(True, alpha=0.3)
        ax2.set_ylabel("v*_n [m/s]")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")

    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        if not self.control_data:
            print("No control_data to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts_full = np.array([d["t"] for d in self.control_data])
        t_full  = ts_full - ts_full[0]

        mask = np.array([("obj_break" in d) for d in self.control_data], dtype=bool)
        if not np.any(mask):
            print("No obj_break logged; ensure you added the logging after solve.")
            return

        impL  = np.array([d["obj_break"]["imp_L"]   for d in self.control_data if "obj_break" in d])
        impR  = np.array([d["obj_break"]["imp_R"]   for d in self.control_data if "obj_break" in d])
        grL   = np.array([d["obj_break"]["grasp_L"] for d in self.control_data if "obj_break" in d])
        grR   = np.array([d["obj_break"]["grasp_R"] for d in self.control_data if "obj_break" in d])
        reg   = np.array([d["obj_break"]["reg"]     for d in self.control_data if "obj_break" in d])
        total = np.array([d["obj_break"]["total"]   for d in self.control_data if "obj_break" in d])
        t_obj = t_full[mask]

        status = [d.get("status", "") for d in self.control_data]
        def classify_status(s):
            s_lower = (s or "").lower()
            if s_lower == "optimal": return 0
            if s_lower in ("optimal_inaccurate","user_limit","max_iters_reached",
                           "iteration_limit_reached","user_limit_reached"): return 1
            if any(key in s_lower for key in [
                "infeasible","unbounded","solver_error","error",
                "dual_infeasible","primal_infeasible",
            ]): return 2
            return 2
        class_vals = np.array([classify_status(s) for s in status], dtype=float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        ax1.plot(t_obj, total, label="total", linewidth=2)
        ax1.plot(t_obj, impL,  label="impedance L")
        ax1.plot(t_obj, impR,  label="impedance R")
        ax1.plot(t_obj, grL,   label="grasp L")
        ax1.plot(t_obj, grR,   label="grasp R")
        ax1.plot(t_obj, reg,   label="regularizer")
        ax1.set_ylabel("objective value"); ax1.grid(True, alpha=0.3); ax1.legend(loc="best", fontsize=8)
        ax1.set_title("QP objective breakdown", fontsize=11)

        ax2.step(t_full, class_vals, where="post", linewidth=2, color="black", label="severity")
        ax2.axhspan(-0.5, 0.5,  facecolor="green",  alpha=0.08)
        ax2.axhspan(0.5, 1.5,   facecolor="yellow", alpha=0.10)
        ax2.axhspan(1.5, 2.5,   facecolor="red",    alpha=0.08)
        ax2.set_yticks([0, 1, 2]); ax2.set_yticklabels(["GOOD", "WARN", "BAD"], fontsize=9)
        ax2.set_ylim(-0.5, 2.5); ax2.set_xlabel("time [s]"); ax2.set_title("QP solve health", fontsize=11)
        ax2.grid(True, axis="x", alpha=0.3)

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fname}")


# =============================================================
# Example main
# =============================================================
if __name__ == "__main__":
    Hz = 60
    K = np.diag([600, 600, 600, 1, 1, 1])

    robotL = URImpedanceController("192.168.1.33", K=K)
    robotR = URImpedanceController("192.168.1.66", K=K)

    # Weighting matrices
    W_imp = diag6([5e4, 5e4, 5e4, 5e3, 5e3, 5e3])
    W_grasp = diag6([0, 0, 0, 0.0, 0.0, 0.0])
    lam_reg = 1e-4

    # Common trajectory (world deltas) → both arms, starts after 3s
    trajectory = "motion_planner/trajectories/lift_100.npz"

    ctrl = DualArmImpedanceAdmittanceQP(
        robotL, robotR, Hz=Hz,
        W_imp=W_imp, W_grasp=W_grasp, lam_reg=lam_reg,
        admittance_gain=3e-4, v_max=0.1, F_n_star=25.0,
        trajectory_path=trajectory,   # uses your format & init rule
        ref_start_delay_s=3.0         # first 3s: grasp only
    )

    try:
        robotL.moveJ([-2.72771532, -1.40769446, 2.81887228, -3.01955523, -1.6224683, 2.31350756])
        robotL.wait_for_commands()

        robotL.wait_for_commands(); robotR.wait_for_commands()
        robotL.go_home(); robotR.go_home()
        robotL.wait_for_commands(); robotR.wait_for_commands(); robotL.wait_until_done(); robotR.wait_until_done()

        robotL.go_to_approach(); robotR.go_to_approach()
        robotL.wait_for_commands(); robotR.wait_for_commands(); robotL.wait_until_done(); robotR.wait_until_done()

        # Run for 20 seconds total. Auto-plot on exit.
        ctrl.run(timeout_s=15.0)
        ctrl.plot_taskspace()
        ctrl.plot_jointspace()
        ctrl.plot_force_tracking()
        ctrl.plot_qp_objective()


    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            robotL.disconnect(); robotR.disconnect()
        except Exception:
            pass
        print("Robots disconnected.")
