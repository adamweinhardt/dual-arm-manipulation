import numpy as np
import time
import threading
import cvxpy as cp
import pinocchio as pin
from scipy.spatial.transform import Rotation as RR
import matplotlib.pyplot as plt
import datetime
import os

from control.pid_ff_controller import URForceController
from utils.utils import _freeze_sparsity


class DualArmGraspingQP:
    def __init__(self, robotL, robotR, Hz):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz

        self.log_every_s = 1.0
        self.control_data = []

        self.joint_pose_limit  = np.deg2rad(360)
        self.joint_speed_limit = np.deg2rad(180)

        # perf
        self._ctrl_start_wall = None
        self._last_log_wall = None
        self._win_iters = 0
        self._win_loop_time = 0.0
        self._win_solver_time = 0.0
        self._win_deadline_miss = 0
        self._total_iters = 0
        self._total_solver_time = 0.0
        self._total_deadline_miss = 0

    # ---------------------- build QP ----------------------
    def build_qp(self, dt):
        n = 6
        self.qdot_L = cp.Variable(n, name="qdot_L")
        self.qdot_R = cp.Variable(n, name="qdot_R")

        self.J_L_p = cp.Parameter((6, n), name="J_L")
        self.J_R_p = cp.Parameter((6, n), name="J_R")
        self.q_L_p = cp.Parameter(n, name="q_L")
        self.q_R_p = cp.Parameter(n, name="q_R")
        self.xdot_star_L = cp.Parameter(6, name="xdot_star_L")
        self.xdot_star_R = cp.Parameter(6, name="xdot_star_R")

        dt_c = cp.Constant(float(dt))

        # objective: track each TCP desired velocity
        obj_L = cp.sum_squares(self.J_L_p @ self.qdot_L - self.xdot_star_L)
        obj_R = cp.sum_squares(self.J_R_p @ self.qdot_R - self.xdot_star_R)
        obj_reg = 1e-4 * (cp.sum_squares(self.qdot_L) + cp.sum_squares(self.qdot_R))
        obj = obj_L + obj_R + obj_reg

        q_pos_lim = self.joint_pose_limit
        q_vel_lim = self.joint_speed_limit
        q_next_L = self.q_L_p + self.qdot_L * dt_c
        q_next_R = self.q_R_p + self.qdot_R * dt_c

        cons = [
            -q_pos_lim <= q_next_L, q_next_L <= q_pos_lim,
            -q_pos_lim <= q_next_R, q_next_R <= q_pos_lim,
            -q_vel_lim <= self.qdot_L, self.qdot_L <= q_vel_lim,
            -q_vel_lim <= self.qdot_R, self.qdot_R <= q_vel_lim,
        ]

        self.qp = cp.Problem(cp.Minimize(obj), cons)
        self.qp_kwargs = dict(
            eps_abs=3e-6,
            eps_rel=3e-6,
            alpha=1.6,
            max_iter=10000,
            adaptive_rho=True,
            adaptive_rho_interval=20,
            polish=True,
            check_termination=10,
            warm_start=True,
        )

    # ---------------------- control loop ----------------------
    def run(self, F_n_star=15.0, k_f=3e-4, v_max=0.02):
        dt = 1.0 / self.Hz
        self.build_qp(dt)
        self.control_stop = threading.Event()
        i = 0

        self._ctrl_start_wall = time.perf_counter()
        self._last_log_wall = time.perf_counter()
        print(f"[GRASP DUAL] starting | F*_n={F_n_star} N, k_f={k_f}, vmax={v_max}")

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                # --- state LEFT ---
                state_L = self.robotL.get_state()
                q_L = self.robotL.get_q()
                R_L = RR.from_rotvec(state_L["pose"][3:6]).as_matrix()
                n_L = -R_L[:, 1]  # +Y_TCP axis
                F_L = np.array(state_L["force_world"][:3])
                F_n_L = float(n_L @ F_L)
                J_L = self.robotL.get_J(q_L)

                # --- state RIGHT ---
                state_R = self.robotR.get_state()
                q_R = self.robotR.get_q()
                R_R = RR.from_rotvec(state_R["pose"][3:6]).as_matrix()
                n_R = -R_R[:, 1]
                F_R = np.array(state_R["force_world"][:3])
                F_n_R = -float(n_R @ F_R)
                J_R = self.robotR.get_J(q_R)

                # --- admittance ---
                v_n_L_star = np.clip(k_f * (F_n_star - F_n_L), -v_max, v_max)
                v_n_R_star = np.clip(k_f * (F_n_star - F_n_R), -v_max, v_max)
                v_star_L = v_n_L_star * n_L
                v_star_R = v_n_R_star * n_R

                xdot_star_L = np.hstack([v_star_L, np.zeros(3)])
                xdot_star_R = np.hstack([v_star_R, np.zeros(3)])

                # --- feed QP ---
                self.J_L_p.value = _freeze_sparsity(J_L)
                self.J_R_p.value = _freeze_sparsity(J_R)
                self.q_L_p.value = q_L
                self.q_R_p.value = q_R
                self.xdot_star_L.value = xdot_star_L
                self.xdot_star_R.value = xdot_star_R

                # --- solve ---
                _t0 = time.perf_counter()
                self.qp.solve(solver=cp.OSQP, **self.qp_kwargs)
                solve_dt = time.perf_counter() - _t0
                self._win_solver_time += solve_dt
                self._total_solver_time += solve_dt

                # extract
                if self.qp.status in ("optimal", "optimal_inaccurate"):
                    qdot_L_cmd = np.asarray(self.qdot_L.value).flatten()
                    qdot_R_cmd = np.asarray(self.qdot_R.value).flatten()
                else:
                    qdot_L_cmd = np.zeros(6)
                    qdot_R_cmd = np.zeros(6)

                # send
                self.robotL.speedJ(qdot_L_cmd.tolist(), dt)
                self.robotR.speedJ(qdot_R_cmd.tolist(), dt)

                # log
                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": self.qp.status,
                    "obj": self.qp.value,
                    "F_n_L": F_n_L,
                    "F_n_R": F_n_R,
                    "F_n_star": F_n_star,
                    "v_n_L_star": v_n_L_star,
                    "v_n_R_star": v_n_R_star,
                    "q_L": q_L,
                    "q_R": q_R,
                    "qdot_L_cmd": qdot_L_cmd,
                    "qdot_R_cmd": qdot_R_cmd,
                })

                # perf
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
                    print(f"[GRASP DUAL] {avg_hz:6.2f} Hz | solver {avg_solver_ms:6.2f} ms | "
                          f"miss {miss_pct:4.1f}% | F_L={F_n_L:6.2f} F_R={F_n_R:6.2f}")
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
        print(f"[GRASP DUAL SUMMARY] Ran {self._total_iters} iters @ "
              f"{self._total_iters/total_time:.1f} Hz")

    # ---------------------- plotting ----------------------
    def plot_force_profile(self, title_prefix="DualGrasp_Forces"):
        if not self.control_data:
            print("No control_data to plot.")
            return
        os.makedirs("plots", exist_ok=True)
        ts = np.array([d["t"] for d in self.control_data])
        t = ts - ts[0]
        F_n_L = np.array([d["F_n_L"] for d in self.control_data])
        F_n_R = np.array([d["F_n_R"] for d in self.control_data])
        F_n_star = np.array([d["F_n_star"] for d in self.control_data])
        v_n_L = np.array([d["v_n_L_star"] for d in self.control_data])
        v_n_R = np.array([d["v_n_R_star"] for d in self.control_data])
        obj = np.array([d["obj"] for d in self.control_data])

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{title_prefix} – {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

        ax = axes[0]
        ax.plot(t, F_n_L, label="F_n_L meas")
        ax.plot(t, F_n_R, label="F_n_R meas")
        ax.plot(t, F_n_star, "--", color="black", label="F*_n ref")
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("Normal Force vs Ref")

        ax = axes[1]
        ax.plot(t, v_n_L, label="v*_n L")
        ax.plot(t, v_n_R, label="v*_n R")
        ax.set_ylabel("TCP Y vel [m/s]")
        ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("Admittance Command Velocity")

        ax = axes[2]
        ax.plot(t, obj, label="QP objective", color="tab:orange")
        ax.set_ylabel("Objective"); ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3); ax.legend()

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        fname = f"plots/{title_prefix.lower()}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")


class URImpedanceController(URForceController):
    def __init__(self, ip):
        super().__init__(ip)
        self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
        self.pin_data = self.pin_model.createData()
        self.pin_frame_id = self.pin_model.getFrameId("tool0")

    def get_q(self):
        return np.array(self.rtde_receive.getActualQ(), dtype=float)

    def get_J(self, q):
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data,
            self.pin_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J


# --------------------------------------------------------------
# Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    robotL = URImpedanceController("192.168.1.33")
    robotR = URImpedanceController("192.168.1.66")

    Hz = 50
    ctrl = DualArmGraspingQP(robotL, robotR, Hz)

    try:
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

        robotL.wait_for_commands(); robotR.wait_for_commands()
        robotL.go_to_approach(); robotR.go_to_approach()
        robotL.wait_until_done(); robotR.wait_until_done()

        ctrl.run(F_n_star=25.0, k_f=3e-4, v_max=0.1)
        ctrl.plot_force_profile()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        robotL.disconnect(); robotR.disconnect()
        print("Robots disconnected.")
