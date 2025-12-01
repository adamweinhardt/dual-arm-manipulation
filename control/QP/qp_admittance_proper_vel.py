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
        v_max=0.5,
        W_adm_L=None,
        W_adm_R=None,
        lambda_reg=1e-6,
    ):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = float(Hz)

        # Grasp normals in BASE frame
        self.normal_L = self.robotL.world_vector_2_robot(self.robotL.get_grasping_data()[2])
        self.normal_R = self.robotR.world_vector_2_robot(self.robotR.get_grasping_data()[2])
        self.grasping_point_L = self.robotL.world_point_2_robot(self.robotL.get_grasping_data()[0])
        self.grasping_point_R = self.robotR.world_point_2_robot(self.robotR.get_grasping_data()[0])

        # Admittance parameters (shared for both arms)
        self.M_a_L = float(M_a); self.D_a_L = float(D_a); self.K_a_L = float(K_a)
        self.M_a_R = float(M_a); self.D_a_R = float(D_a); self.K_a_R = float(K_a)

        # Admittance states (normal direction)
        self.x_n_L = 0.0; self.v_n_L = 0.0
        self.x_n_R = 0.0; self.v_n_R = 0.0

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
        self.v_n_max = float(v_max)

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

    # ---------------------- build QP (accel var, velocity tracking) ----------------------
    def build_qp(self, dt):
        n = 6
        self.q_ddot_L = cp.Variable(n, name="q_ddot_L")
        self.q_ddot_R = cp.Variable(n, name="q_ddot_R")

        # Left arm parameters
        self.J_L_p = cp.Parameter((6, n), name="J_L")
        self.q_L_p = cp.Parameter(n, name="q_L")
        self.q_dot_L_p = cp.Parameter(n, name="q_dot_L")
        self.v_des_L_p = cp.Parameter(6, name="v_des_L")
        self.W_adm_L_p = cp.Parameter((6, 6), name="W_adm_L")

        # Right arm parameters
        self.J_R_p = cp.Parameter((6, n), name="J_R")
        self.q_R_p = cp.Parameter(n, name="q_R")
        self.q_dot_R_p = cp.Parameter(n, name="q_dot_R")
        self.v_des_R_p = cp.Parameter(6, name="v_des_R")
        self.W_adm_R_p = cp.Parameter((6, 6), name="W_adm_R")

        dt_c = cp.Constant(float(dt))
        dt2_c = cp.Constant(float(0.5 * dt * dt))

        # Next-step kinematics (expressions)
        q_dot_next_L = self.q_dot_L_p + self.q_ddot_L * dt_c
        q_dot_next_R = self.q_dot_R_p + self.q_ddot_R * dt_c

        q_next_L = self.q_L_p + self.q_dot_L_p * dt_c + self.q_ddot_L * dt2_c
        q_next_R = self.q_R_p + self.q_dot_R_p * dt_c + self.q_ddot_R * dt2_c

        # Task-space velocity tracking error at next step
        e_vel_L = self.J_L_p @ q_dot_next_L - self.v_des_L_p
        e_vel_R = self.J_R_p @ q_dot_next_R - self.v_des_R_p

        obj = (
            cp.sum_squares(self.W_adm_L_p @ e_vel_L)
            + cp.sum_squares(self.W_adm_R_p @ e_vel_R)
            + self.lambda_reg * (cp.sum_squares(self.q_ddot_L) + cp.sum_squares(self.q_ddot_R))
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
    def run(self, ref_force):
        time.sleep(1) 
        dt = 1.0 / self.Hz
        self.build_qp(dt)
        self.control_stop = threading.Event()
        self.robotL.rtde_control.zeroFtSensor()
        self.robotR.rtde_control.zeroFtSensor()

        # Reset admittance states
        self.x_n_L = 0.0
        self.v_n_L = 0.0
        self.x_n_R = 0.0
        self.v_n_R = 0.0

        i = 0
        self._ctrl_start_wall = time.perf_counter()
        self._last_log_wall = time.perf_counter()

        print(
            f"[DUAL ADMITTANCE a->v QP] start | F*_n={ref_force} N | "
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

                # TCP position & linear velocity in base frame
                p_L = np.array(state_L["gripper_base"][:3], dtype=float)
                v_L = np.array(state_L["speed"][:3], dtype=float)

                # initialize grasping point if not set
                if getattr(self, "grasping_point_L", None) is None:
                    self.grasping_point_L = p_L.copy()

                # signed distance from grasp plane along normal
                #d_L = np.linalg.norm(p_L - self.grasping_point_L)
                d_L = float((p_L - self.grasping_point_L) @ n_L)

                # --- RIGHT state ---
                state_R = self.robotR.get_state()
                q_R = self.robotR.get_q()
                q_dot_R = self.robotR.get_qdot()
                n_R = self.normal_R
                F_R_vec = np.array(state_R["filtered_force"][:3], dtype=float)
                J_R = self.robotR.get_J(q_R)

                p_R = np.array(state_R["gripper_base"][:3], dtype=float)
                v_R = np.array(state_R["speed"][:3], dtype=float)

                if getattr(self, "grasping_point_R", None) is None:
                    self.grasping_point_R = p_R.copy()

                #d_R = np.linalg.norm(p_R - self.grasping_point_R)
                d_R = float((p_R - self.grasping_point_R) @ n_R)

                # --- normal forces & errors ---
                F_n_L = float(n_L @ F_L_vec)
                F_n_R = float(n_R @ F_R_vec)
                e_n_L = ref_force - F_n_L
                e_n_R = ref_force - F_n_R

                # --- 1D mass–spring–damper admittance anchored at grasp point ---
                # M * a_n + D * v_n + K * x_n = e_n, with x_n = geometric distance d_{L/R}
                a_n_L = (e_n_L - self.D_a_L * self.v_n_L - self.K_a_L * d_L) / self.M_a_L
                self.v_n_L += a_n_L * dt
                self.x_n_L = d_L  # x_n is now true distance to grasp plane along normal

                a_n_R = (e_n_R - self.D_a_R * self.v_n_R - self.K_a_R * d_R) / self.M_a_R
                self.v_n_R += a_n_R * dt
                self.x_n_R = d_R

                self.v_n_L = np.clip(self.v_n_L, -self.v_n_max, self.v_n_max)
                self.v_n_R = np.clip(self.v_n_R, -self.v_n_max, self.v_n_max)

                # Desired TCP velocities (linear) along -n (force too low -> move into the object)
                v_des_L = np.hstack([(self.v_n_L) * -n_L, np.zeros(3)])
                v_des_R = np.hstack([(self.v_n_R) * -n_R, np.zeros(3)])

                # --- feed QP parameters ---
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.q_L_p.value = q_L
                self.q_dot_L_p.value = q_dot_L
                self.v_des_L_p.value = v_des_L
                self.W_adm_L_p.value = self.W_adm_L

                self.J_R_p.value = _freeze_sparsity(J_R)
                self.q_R_p.value = q_R
                self.q_dot_R_p.value = q_dot_R
                self.v_des_R_p.value = v_des_R
                self.W_adm_R_p.value = self.W_adm_R

                # --- solve QP ---
                _t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                solve_dt = time.perf_counter() - _t0
                self._win_solver_time += solve_dt
                self._total_solver_time += solve_dt

                status = (self.qp.status or "").lower()
                ok = status in ("optimal", "optimal_inaccurate")

                # --- extract commands (integrated velocity) ---
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

                if i % 50 == 0:
                    dp_L = p_L - self.grasping_point_L
                    dp_R = p_R - self.grasping_point_R

                    # normalize normals for scalar projections
                    nL = n_L / (np.linalg.norm(n_L) + 1e-12)
                    nR = n_R / (np.linalg.norm(n_R) + 1e-12)

                    v_des_L_lin = v_des_L[:3]
                    v_des_R_lin = v_des_R[:3]

                    v_L_n = float(v_L @ nL)
                    v_R_n = float(v_R @ nR)
                    v_des_L_n = float(v_des_L_lin @ nL)
                    v_des_R_n = float(v_des_R_lin @ nR)

                    d_proj_L = float(dp_L @ nL)
                    d_proj_R = float(dp_R @ nR)

                    print(f"\n[DBG {i}] ================================")

                    print("LEFT ARM – BASE FRAME:")
                    print(f"  n (normal)        = {n_L}")
                    print(f"  p_tcp             = {p_L}")
                    print(f"  p_grasp           = {self.grasping_point_L}")
                    print(f"  dp = p_tcp - p_g  = {dp_L}")
                    print(f"  d_L (used)        = {d_L: .4f} m  (along -n)")
                    print(f"  d_proj_n          = {d_proj_L: .4f} m (dp·n)")
                    print(f"  F_vec             = {F_L_vec}")
                    print(f"  F_n               = {F_n_L: .4f} N | F_ref = {ref_force:.2f} | e_F = {e_n_L: .4f}")
                    print(f"  v_tcp (base)      = {v_L}")
                    print(f"  v_tcp·n           = {v_L_n: .4f} m/s")
                    print(f"  v_des (base)      = {v_des_L}")
                    print(f"  v_des·n           = {v_des_L_n: .4f} m/s")
                    print(f"  v_n_state         = {self.v_n_L: .4f} m/s")
                    print(f"  x_n_state         = {self.x_n_L: .4f} m")

                    print("\nRIGHT ARM – BASE FRAME:")
                    print(f"  n (normal)        = {n_R}")
                    print(f"  p_tcp             = {p_R}")
                    print(f"  p_grasp           = {self.grasping_point_R}")
                    print(f"  dp = p_tcp - p_g  = {dp_R}")
                    print(f"  d_R (used)        = {d_R: .4f} m  (along -n)")
                    print(f"  d_proj_n          = {d_proj_R: .4f} m (dp·n)")
                    print(f"  F_vec             = {F_R_vec}")
                    print(f"  F_n               = {F_n_R: .4f} N | F_ref = {ref_force:.2f} | e_F = {e_n_R: .4f}")
                    print(f"  v_tcp (base)      = {v_R}")
                    print(f"  v_tcp·n           = {v_R_n: .4f} m/s")
                    print(f"  v_des (base)      = {v_des_R}")
                    print(f"  v_des·n           = {v_des_R_n: .4f} m/s")
                    print(f"  v_n_state         = {self.v_n_R: .4f} m/s")
                    print(f"  x_n_state         = {self.x_n_R: .4f} m")

                    print("============================================")



                # --- log data ---
                self.control_data.append(
                    {
                        "t": time.time(),
                        "i": i,
                        "status": status,
                        "obj": self.qp.value,

                        # forces
                        "F_n_L": F_n_L,
                        "F_n_R": F_n_R,
                        "ref_force": ref_force,

                        # admittance states (normal 1D)
                        "x_n_L": self.x_n_L,
                        "x_n_R": self.x_n_R,
                        "v_n_L": self.v_n_L,
                        "v_n_R": self.v_n_R,
                        "a_n_L": a_n_L,
                        "a_n_R": a_n_R,

                        # joint stuff
                        "q_L": q_L,
                        "q_R": q_R,
                        "q_dot_L_meas": q_dot_L,
                        "q_dot_R_meas": q_dot_R,
                        "q_ddot_L_cmd": q_ddot_L_cmd,
                        "q_ddot_R_cmd": q_ddot_R_cmd,
                        "q_dot_L_cmd": q_dot_L_cmd,
                        "q_dot_R_cmd": q_dot_R_cmd,

                        # --- base-frame metrics you care about ---
                        "p_L": p_L,
                        "p_R": p_R,
                        "grasp_L": self.grasping_point_L,
                        "grasp_R": self.grasping_point_R,
                        "d_L": d_L,
                        "d_R": d_R,
                        "v_tcp_L": v_L,
                        "v_tcp_R": v_R,
                        "v_des_L": v_des_L,
                        "v_des_R": v_des_R,
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
                        f"[DUAL ADMITTANCE a->v QP] {avg_hz:6.2f} Hz | "
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
            f"[DUAL ADMITTANCE a->v QP SUMMARY] Ran {self._total_iters} iters @ "
            f"{self._total_iters / total_time:.1f} Hz"
        )


    # ---------------------- plotting ----------------------
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



class URImpedanceController(URForceController):
    """
    Minimal UR wrapper giving joint state and Pinocchio Jacobians,
    suitable for use with DualArmAdmittanceAccelQP (velocity-tracking).
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
        # Not used in this velocity-tracking formulation, kept for compatibility
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

    Hz = 50

    # Heavier weights on linear tracking; de-emphasize orientation
    W_adm = np.diag([1, 1, 1, 1e6, 1e6, 1e6])
    lambda_reg = 1e-6
    v_max = 0.05  # m/s

    M = 27
    K = 300
    D = 2400.0 #2 * np.sqrt(M * K)

    ctrl = DualArmAdmittanceAccelQP(
        robotL=robotL,
        robotR=robotR,
        Hz=Hz,
        M_a=M,
        D_a=D,
        K_a=K,
        v_max=v_max,
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

        robotL.go_to_approach()
        robotR.go_to_approach()
        robotL.wait_for_commands()
        robotR.wait_for_commands()
        robotL.wait_until_done()
        robotR.wait_until_done()

        # Run QP (accel variable, velocity tracking)
        ctrl.run(ref_force=25.0)

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
