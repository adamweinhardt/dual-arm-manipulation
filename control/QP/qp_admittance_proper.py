#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import threading
import cvxpy as cp
import pinocchio as pin
from scipy.spatial.transform import Rotation as RR
import matplotlib.pyplot as plt
import datetime
import os

# project deps
from control.PID.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity  # adjust import if needed


class DualArmAdmittanceAccelQP:
    def __init__(
        self,
        robotL,
        robotR,
        Hz,
        M_a=3.0,
        D_a=40.0,
        K_a=0.0,
        W_adm_L=None,
        W_adm_R=None,
        lambda_reg=1e-6,
    ):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = float(Hz)

        # Grasp normals in each robot base frame
        # get_grasping_data()[2] assumed to be world normal; convert to robot frame
        self.normal_L = self.robotL.world_vector_2_robot(self.robotL.get_grasping_data()[2])
        self.normal_R = self.robotR.world_vector_2_robot(self.robotR.get_grasping_data()[2])

        # Normalize normals for safety
        self.normal_L = self.normal_L / np.linalg.norm(self.normal_L)
        self.normal_R = self.normal_R / np.linalg.norm(self.normal_R)

        # Admittance parameters (shared for both arms)
        self.M_a_L = float(M_a)
        self.D_a_L = float(D_a)
        self.K_a_L = float(K_a)
        self.M_a_R = float(M_a)
        self.D_a_R = float(D_a)
        self.K_a_R = float(K_a)

        # Admittance states (normal direction)
        self.x_n_L = 0.0  # virtual displacement along n_L
        self.v_n_L = 0.0  # virtual velocity along n_L
        self.x_n_R = 0.0
        self.v_n_R = 0.0

        # Logging
        self.log_every_s = 1.0
        self.control_data = []

        # Limits (tune to your UR model)
        self.joint_pose_limit = np.deg2rad(360.0)
        self.joint_speed_limit = np.deg2rad(180.0)
        self.joint_accel_limit = np.deg2rad(120.0)

        # QP weights
        self.lambda_reg = float(lambda_reg)
        self.W_adm_L = np.asarray(W_adm_L, dtype=float) if W_adm_L is not None else np.eye(6)
        self.W_adm_R = np.asarray(W_adm_R, dtype=float) if W_adm_R is not None else np.eye(6)

        # Perf metrics
        self._ctrl_start_wall = None
        self._last_log_wall = None
        self._win_iters = 0
        self._win_loop_time = 0.0
        self._win_solver_time = 0.0
        self._win_deadline_miss = 0
        self._total_iters = 0
        self._total_solver_time = 0.0
        self._total_deadline_miss = 0

        self.qp = None
        self.qp_kwargs = None

    # ---------------------- build QP (acceleration-level) ----------------------
    def build_qp(self, dt):
        n = 6
        self.q_ddot_L = cp.Variable(n, name="q_ddot_L")
        self.q_ddot_R = cp.Variable(n, name="q_ddot_R")

        # Left arm parameters
        self.J_L_p = cp.Parameter((6, n), name="J_L")
        self.J_dot_L_p = cp.Parameter((6, n), name="J_dot_L")
        self.q_L_p = cp.Parameter(n, name="q_L")
        self.q_dot_L_p = cp.Parameter(n, name="q_dot_L")
        self.a_des_L_p = cp.Parameter(6, name="a_des_L")
        self.W_adm_L_p = cp.Parameter((6, 6), name="W_adm_L")

        # Right arm parameters
        self.J_R_p = cp.Parameter((6, n), name="J_R")
        self.J_dot_R_p = cp.Parameter((6, n), name="J_dot_R")
        self.q_R_p = cp.Parameter(n, name="q_R")
        self.q_dot_R_p = cp.Parameter(n, name="q_dot_R")
        self.a_des_R_p = cp.Parameter(6, name="a_des_R")
        self.W_adm_R_p = cp.Parameter((6, 6), name="W_adm_R")

        dt_c = cp.Constant(float(dt))
        dt2_c = cp.Constant(float(0.5 * dt * dt))

        # Left dynamics
        q_next_L = self.q_L_p + self.q_dot_L_p * dt_c + self.q_ddot_L * dt2_c
        q_dot_next_L = self.q_dot_L_p + self.q_ddot_L * dt_c
        e_imp_L = self.J_L_p @ self.q_ddot_L + self.J_dot_L_p @ self.q_dot_L_p - self.a_des_L_p

        # Right dynamics
        q_next_R = self.q_R_p + self.q_dot_R_p * dt_c + self.q_ddot_R * dt2_c
        q_dot_next_R = self.q_dot_R_p + self.q_ddot_R * dt_c
        e_imp_R = self.J_R_p @ self.q_ddot_R + self.J_dot_R_p @ self.q_dot_R_p - self.a_des_R_p

        obj = (
            cp.sum_squares(self.W_adm_L_p @ e_imp_L)
            + cp.sum_squares(self.W_adm_R_p @ e_imp_R)
            + self.lambda_reg * cp.sum_squares(self.q_ddot_L)
            + self.lambda_reg * cp.sum_squares(self.q_ddot_R)
        )

        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_acc_lim = self.joint_accel_limit

        cons = [
            -q_pos_lim <= q_next_L, q_next_L <= q_pos_lim,
            -q_vel_lim <= q_dot_next_L, q_dot_next_L <= q_vel_lim,
            -q_acc_lim <= self.q_ddot_L, self.q_ddot_L <= q_acc_lim,

            -q_pos_lim <= q_next_R, q_next_R <= q_pos_lim,
            -q_vel_lim <= q_dot_next_R, q_dot_next_R <= q_vel_lim,
            -q_acc_lim <= self.q_ddot_R, self.q_ddot_R <= q_acc_lim,
        ]

        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(
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

    # ---------------------- control loop ----------------------
    def run(self, ref_force=25.0, M_a=None, D_a=None, K_a=None):
        """
        ref_force : desired normal force magnitude [N]
        M_a, D_a, K_a : optional overrides for admittance parameters (same for both arms)
        """
        dt = 1.0 / self.Hz
        self.build_qp(dt)
        self.control_stop = threading.Event()

        # Override admittance params for this run if provided
        if M_a is not None:
            self.M_a_L = self.M_a_R = float(M_a)
        if D_a is not None:
            self.D_a_L = self.D_a_R = float(D_a)
        if K_a is not None:
            self.K_a_L = self.K_a_R = float(K_a)

        # Reset admittance states
        self.x_n_L = 0.0
        self.v_n_L = 0.0
        self.x_n_R = 0.0
        self.v_n_R = 0.0

        i = 0
        self._ctrl_start_wall = time.perf_counter()
        self._last_log_wall = time.perf_counter()

        print(
            f"[DUAL ADMITTANCE a-QP] start | F*_n={ref_force} N | "
            f"M={self.M_a_L}, D={self.D_a_L}, K={self.K_a_L}"
        )

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                # --- LEFT state ---
                state_L = self.robotL.get_state()
                q_L = self.robotL.get_q()
                q_dot_L = self.robotL.get_qdot()
                n_L = self.normal_L
                F_L_vec = np.array(state_L["filtered_force"][:3], dtype=float)
                J_L = self.robotL.get_J(q_L)
                J_dot_L = self.robotL.get_Jdot(q_L, q_dot_L)

                # --- RIGHT state ---
                state_R = self.robotR.get_state()
                q_R = self.robotR.get_q()
                q_dot_R = self.robotR.get_qdot()
                n_R = self.normal_R
                F_R_vec = np.array(state_R["filtered_force"][:3], dtype=float)
                J_R = self.robotR.get_J(q_R)
                J_dot_R = self.robotR.get_Jdot(q_R, q_dot_R)

                # --- force references (vector) ---
                F_L_ref_vec = ref_force * n_L
                F_R_ref_vec = ref_force * n_R

                # vector force errors
                e_F_L_vec = F_L_ref_vec - F_L_vec
                e_F_R_vec = F_R_ref_vec - F_R_vec

                # scalar normal force measurements and errors
                F_n_L = float(n_L @ F_L_vec)
                F_n_R = float(n_R @ F_R_vec)
                e_n_L = float(e_F_L_vec @ n_L)
                e_n_R = float(e_F_R_vec @ n_R)

                # --- SECOND-ORDER ADMITTANCE (1D along normal) ---
                # LEFT: M * a + D * v + K * x = e
                a_n_L = (e_n_L - self.D_a_L * self.v_n_L - self.K_a_L * self.x_n_L) / self.M_a_L
                self.v_n_L += a_n_L * dt
                self.x_n_L += self.v_n_L * dt

                # RIGHT
                a_n_R = (e_n_R - self.D_a_R * self.v_n_R - self.K_a_R * self.x_n_R) / self.M_a_R
                self.v_n_R += a_n_R * dt
                self.x_n_R += self.v_n_R * dt

                # Task-space desired accelerations (linear part only)
                a_des_L = np.hstack([a_n_L * -n_L, np.zeros(3)])
                a_des_R = np.hstack([a_n_R * -n_R, np.zeros(3)])

                # --- feed QP parameters ---
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.J_dot_L_p.value = _freeze_sparsity(J_dot_L)
                self.q_L_p.value = q_L
                self.q_dot_L_p.value = q_dot_L
                self.a_des_L_p.value = a_des_L
                self.W_adm_L_p.value = self.W_adm_L

                self.J_R_p.value = _freeze_sparsity(J_R)
                self.J_dot_R_p.value = _freeze_sparsity(J_dot_R)
                self.q_R_p.value = q_R
                self.q_dot_R_p.value = q_dot_R
                self.a_des_R_p.value = a_des_R
                self.W_adm_R_p.value = self.W_adm_R

                # --- solve QP ---
                _t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                solve_dt = time.perf_counter() - _t0
                self._win_solver_time += solve_dt
                self._total_solver_time += solve_dt

                status = (self.qp.status or "").lower()
                ok = status in ("optimal", "optimal_inaccurate")

                # --- extract commands ---
                if ok:
                    q_ddot_L_cmd = np.asarray(self.q_ddot_L.value).flatten()
                    q_ddot_R_cmd = np.asarray(self.q_ddot_R.value).flatten()
                    q_dot_L_cmd = q_dot_L + q_ddot_L_cmd * dt
                    q_dot_R_cmd = q_dot_R + q_ddot_R_cmd * dt
                else:
                    q_ddot_L_cmd = np.zeros_like(q_L)
                    q_ddot_R_cmd = np.zeros_like(q_R)
                    q_dot_L_cmd = q_dot_L
                    q_dot_R_cmd = q_dot_R

                # --- send joint velocity commands ---
                self.robotL.speedJ(q_dot_L_cmd.tolist(), dt)
                self.robotR.speedJ(q_dot_R_cmd.tolist(), dt)

                # ==================== RAW DEBUG ====================
                if i % 50 == 0:
                    # Linear parts
                    aL_lin = a_des_L[:3]
                    aR_lin = a_des_R[:3]

                    # Normalized normals
                    nL = n_L / (np.linalg.norm(n_L) + 1e-12)
                    nR = n_R / (np.linalg.norm(n_R) + 1e-12)

                    # Projections of desired accel along normal
                    a_proj_L = float(np.dot(aL_lin, nL))
                    a_proj_R = float(np.dot(aR_lin, nR))

                    # Measured TCP velocities (base frame)
                    v_ee_L = (J_L @ q_dot_L)[:3]
                    v_ee_R = (J_R @ q_dot_R)[:3]
                    v_proj_L = float(np.dot(v_ee_L, nL))
                    v_proj_R = float(np.dot(v_ee_R, nR))

                    # Force projections and reference error
                    F_proj_L = float(np.dot(F_L_vec, nL))
                    F_proj_R = float(np.dot(F_R_vec, nR))
                    eF_proj_L = float(ref_force - F_proj_L)
                    eF_proj_R = float(ref_force - F_proj_R)

                    # Joint command magnitudes
                    qdd_cmd_L = np.asarray(self.q_ddot_L.value).flatten() if self.q_ddot_L.value is not None else np.zeros_like(q_L)
                    qdd_cmd_R = np.asarray(self.q_ddot_R.value).flatten() if self.q_ddot_R.value is not None else np.zeros_like(q_R)
                    qdot_cmd_L = q_dot_L + qdd_cmd_L * dt
                    qdot_cmd_R = q_dot_R + qdd_cmd_R * dt

                    print("\n[DBG %d]" % i)
                    print("  F_n_L=%.3f  F_n_R=%.3f" % (F_proj_L, F_proj_R))
                    print("  eF_n_L=%.3f  eF_n_R=%.3f" % (eF_proj_L, eF_proj_R))
                    print("  a_des·n  L/R = %.4f / %.4f" % (a_proj_L, a_proj_R))
                    print("  v_ee·n   L/R = %.4f / %.4f" % (v_proj_L, v_proj_R))
                    print("  ||a_des_lin|| L/R = %.4f / %.4f" % (np.linalg.norm(aL_lin), np.linalg.norm(aR_lin)))
                    print("  ||qdd_cmd|| L/R = %.4f / %.4f" % (np.linalg.norm(qdd_cmd_L), np.linalg.norm(qdd_cmd_R)))
                    print("  ||qdot_cmd|| L/R = %.4f / %.4f" % (np.linalg.norm(qdot_cmd_L), np.linalg.norm(qdot_cmd_R)))
                # ================= END RAW DEBUG ====================


                # --- log data ---
                self.control_data.append(
                    {
                        "t": time.time(),
                        "i": i,
                        "status": status,
                        "obj": self.qp.value,
                        "F_n_L": F_n_L,
                        "F_n_R": F_n_R,
                        "ref_force": ref_force,
                        "x_n_L": self.x_n_L,
                        "x_n_R": self.x_n_R,
                        "v_n_L": self.v_n_L,
                        "v_n_R": self.v_n_R,
                        "a_n_L": a_n_L,
                        "a_n_R": a_n_R,
                        "q_L": q_L,
                        "q_R": q_R,
                        "q_dot_L_meas": q_dot_L,
                        "q_dot_R_meas": q_dot_R,
                        "q_ddot_L_cmd": q_ddot_L_cmd,
                        "q_ddot_R_cmd": q_ddot_R_cmd,
                        "q_dot_L_cmd": q_dot_L_cmd,
                        "q_dot_R_cmd": q_dot_R_cmd,
                        "solver_time": solve_dt,
                    }
                )

                # --- perf stats ---
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
                    avg_hz = 1.0 / avg_period if avg_period > 0 else np.nan
                    avg_solver_ms = (self._win_solver_time / self._win_iters) * 1000.0
                    miss_pct = 100.0 * self._win_deadline_miss / self._win_iters
                    print(
                        f"[DUAL ADMITTANCE a-QP] {avg_hz:6.2f} Hz | "
                        f"solver {avg_solver_ms:6.2f} ms | "
                        f"miss {miss_pct:4.1f}% | "
                        f"F_L={F_n_L:6.2f} F_R={F_n_R:6.2f}"
                    )
                    self._win_iters = 0
                    self._win_loop_time = 0.0
                    self._win_solver_time = 0.0
                    self._win_deadline_miss = 0
                    self._last_log_wall = now

                time.sleep(max(0, dt - elapsed))
                i += 1

            except KeyboardInterrupt:
                print("User stop.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                break

        self.robotL.speedStop()
        self.robotR.speedStop()
        total_time = time.perf_counter() - self._ctrl_start_wall
        print(
            f"[DUAL ADMITTANCE a-QP SUMMARY] Ran {self._total_iters} iters @ "
            f"{self._total_iters / total_time:.1f} Hz"
        )

    # ---------------------- plotting ----------------------
    def plot_force_profile(self, title_prefix="DualAdmittanceAccel_Forces"):
        if not self.control_data:
            print("No control_data to plot.")
            return
        os.makedirs("plots", exist_ok=True)
        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]

        F_n_L = np.array([d["F_n_L"] for d in self.control_data])
        F_n_R = np.array([d["F_n_R"] for d in self.control_data])
        ref_force = np.array([d["ref_force"] for d in self.control_data])

        v_n_L = np.array([d["v_n_L"] for d in self.control_data])
        v_n_R = np.array([d["v_n_R"] for d in self.control_data])
        a_n_L = np.array([d["a_n_L"] for d in self.control_data])
        a_n_R = np.array([d["a_n_R"] for d in self.control_data])

        obj = np.array([d["obj"] for d in self.control_data])

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{title_prefix} – {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

        ax = axes[0]
        ax.plot(t, F_n_L, label="F_n_L meas")
        ax.plot(t, F_n_R, label="F_n_R meas")
        ax.plot(t, ref_force, "--", color="black", label="F*_n ref")
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Normal Force vs Ref")

        ax = axes[1]
        ax.plot(t, v_n_L, label="v_n L")
        ax.plot(t, v_n_R, label="v_n R")
        ax.set_ylabel("v_n [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Admittance Normal Velocity (state)")

        ax = axes[2]
        ax.plot(t, a_n_L, label="a_n L")
        ax.plot(t, a_n_R, label="a_n R")
        ax.set_ylabel("a_n [m/s^2]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Admittance Normal Acceleration")

        ax = axes[3]
        ax.plot(t, obj, label="QP objective", color="tab:orange")
        ax.set_ylabel("Objective")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        fname = f"plots/{title_prefix.lower()}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")


class URImpedanceController(URForceController):
    """
    Minimal UR wrapper giving joint state and Pinocchio Jacobians,
    suitable for use with DualArmAdmittanceAccelQP.
    """

    def __init__(self, ip):
        super().__init__(ip)
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
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return self.get_J_pin(J)

    def get_Jdot(self, q, qdot):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, qdot)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model,
            self.pin_data,
            self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return self.get_J_pin(dJ)


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate robots
    robotL = URImpedanceController("192.168.1.33")
    robotR = URImpedanceController("192.168.1.66")

    Hz = 75

    # Simple identity weights for task acceleration tracking
    W_adm = np.diag([1.0, 1.0, 1.0, 1e5, 1e5, 1e5])
    lambda_reg = 0

    ctrl = DualArmAdmittanceAccelQP(
        robotL=robotL,
        robotR=robotR,
        Hz=Hz,
        M_a=3.0,
        D_a=40.0,
        K_a=0.0,
        W_adm_L=W_adm,
        W_adm_R=W_adm,
        lambda_reg=lambda_reg,
    )

    try:
        # Example: move to some start pose, then approach
        robotL.moveJ(
            [-2.72771532, -1.40769446, 2.81887228, -3.01955523, -1.6224683, 2.31350756]
        )
        robotL.wait_for_commands()

        robotR.go_home()
        robotL.go_home()
        robotR.wait_for_commands()
        robotL.wait_for_commands()
        robotR.wait_until_done()
        robotL.wait_until_done()

        robotL.wait_for_commands()
        robotR.wait_for_commands()
        robotL.go_to_approach()
        robotR.go_to_approach()
        robotL.wait_until_done()
        robotR.wait_until_done()

        # Run acceleration-level admittance QP controller
        ctrl.run(ref_force=25.0, M_a=10.0, D_a=50.0, K_a=500.0)

        # Plot simple force profile
        ctrl.plot_force_profile()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            robotL.disconnect()
            robotR.disconnect()
        except Exception:
            pass
        print("Robots disconnected.")
