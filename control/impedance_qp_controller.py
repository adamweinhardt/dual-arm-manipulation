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

        #references
        trajectory_npz=np.load(trajectory)
        self.position_ref = trajectory_npz['position']
        self.velocity_ref = trajectory_npz['linear_velocity']
        self.rotation_ref = trajectory_npz['rotation_matrices']
        self.angular_velocity_ref = trajectory_npz['angular_velocity']

        #limits
        self.joint_pose_limit = np.deg2rad(363)  # +- rad
        self.joint_speed_limit = np.deg2rad(190)  # +- rad/s
        self.joint_accel_limit = np.deg2rad(80)  # +- rad/s^2

        self.control_data = []

    def run(self):
        dt = 1.0 / self.Hz
        self.control_stop = threading.Event()
        i = 0

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:

                # --- References ---
                p_ref = self.position_ref[i]
                v_ref = self.velocity_ref[i]
                R_ref = self.rotation_ref[i]
                w_ref = self.angular_velocity_ref[i]

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

                f_L = self.robotL.get_wrench_desired(D_L, self.robotL.K, p_L, v_L, R_L, w_L, p_ref, v_ref, R_ref, w_ref)

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

                f_R = self.robotR.get_wrench_desired(D_R, self.robotR.K, p_R, v_R, R_R, w_R, p_ref, v_ref, R_ref, w_ref)

                # --- QP Optimization ---
                n_L = q_L.size
                n_R = q_R.size

                # decision variables
                qddot_L = cp.Variable(n_L)
                qddot_R = cp.Variable(n_R)

                # cost function
                xddot_L = J_L @ qddot_L + Jdot_L @ qdot_L
                xddot_R = J_R @ qddot_R + Jdot_R @ qdot_R

                e_l = xddot_L - np.linalg.inv(Lambda_L) @ f_L
                e_r = xddot_R - np.linalg.inv(Lambda_R) @ f_R

                cost = cp.sum_squares(e_l) + cp.sum_squares(e_r)
                constraints = []

                for j in range(n_L):
                    # joint position limits
                    q_next_L = q_L[j] + qdot_L[j] * dt + 0.5 * qddot_L[j] * dt**2
                    q_next_R = q_R[j] + qdot_R[j] * dt + 0.5 * qddot_R[j] * dt**2
                    constraints += [q_next_L <= self.joint_pose_limit, q_next_L >= -self.joint_pose_limit]
                    constraints += [q_next_R <= self.joint_pose_limit, q_next_R >= -self.joint_pose_limit]  
                    # joint speed limits
                    qdot_next_L = qdot_L[j] + qddot_L[j] * dt
                    qdot_next_R = qdot_R[j] + qddot_R[j] * dt
                    constraints += [qdot_next_L <= self.joint_speed_limit, qdot_next_L >= -self.joint_speed_limit]
                    constraints += [qdot_next_R <= self.joint_speed_limit, qdot_next_R >= -self.joint_speed_limit]
                    # joint acceleration limits
                    constraints += [qddot_L[j] <= self.joint_accel_limit, qddot_L[j] >= -self.joint_accel_limit]
                    constraints += [qddot_R[j] <= self.joint_accel_limit, qddot_R[j] >= -self.joint_accel_limit]

                # solve QP
                problem = cp.Problem(cp.Minimize(cost), constraints)
                problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000)

                if problem.status not in ("optimal", "optimal_inaccurate"):
                    qddot_L_sol = np.zeros_like(q_L)
                    qddot_R_sol = np.zeros_like(q_R)
                else:
                    qddot_L_sol = qddot_L.value
                    qddot_R_sol = qddot_R.value

                # integrate
                qdot_L_cmd = qdot_L + qddot_L_sol*dt
                q_L_cmd    = q_L    + qdot_L*dt + 0.5*qddot_L_sol*dt*dt

                qdot_R_cmd = qdot_R + qddot_R_sol*dt
                q_R_cmd    = q_R    + qdot_R*dt + 0.5*qddot_R_sol*dt*dt

                # send commands
                self.robotL.speedJ(qdot_L_cmd.tolist(), dt)
                self.robotR.speedJ(qdot_R_cmd.tolist(), dt)

                i += 1


                # measured task states (use what you already computed)
                tcp_log_L = dict(
                    p=p_L, v=v_L, rvec=r_L, w=w_L,
                    p_ref=p_ref, v_ref=v_ref,
                    rvec_ref=RR.from_matrix(R_ref).as_rotvec(),
                    w_ref=w_ref
                )
                tcp_log_R = dict(
                    p=p_R, v=v_R, rvec=r_R, w=w_R,
                    p_ref=p_ref, v_ref=v_ref,
                    rvec_ref=RR.from_matrix(R_ref).as_rotvec(),
                    w_ref=w_ref
                )

                self.control_data.append({
                    "t": time.time(),
                    "i": i,
                    "status": problem.status,
                    "obj": problem.value,
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

            # Keep loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, dt - elapsed))

        # Clean stop
        try:
            self.robotL.speedStop()
            self.robotR.speedStop()
        except:
            pass

    def _plot_joint_space(self, logs, which="L"):
        """Joint-space plots: q, qdot (meas & cmd), qddot. Returns fig."""
        if not logs:
            return None
        t   = np.array([e["t"] for e in logs]) - logs[0]["t"]
        q   = np.stack([e[f"q_{which}"] for e in logs])
        qd  = np.stack([e[f"qdot_{which}"] for e in logs])
        qdd = np.stack([e[f"qddot_{which}"] for e in logs])
        qd_cmd = np.stack([e[f"qdot_cmd_{which}"] for e in logs])

        fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        axs[0].plot(t, q);       axs[0].set_ylabel(f"q_{which} [rad]")
        axs[1].plot(t, qd);      axs[1].plot(t, qd_cmd, "--"); axs[1].set_ylabel(f"q̇_{which} [rad/s]")
        axs[1].legend(["meas", "cmd"], loc="best")
        axs[2].plot(t, qdd);     axs[2].set_ylabel(f"q̈_{which} [rad/s²]")
        axs[2].set_xlabel("t [s]")
        for ax in axs: ax.grid(True)
        fig.suptitle(f"Joint-space ({which})")
        fig.tight_layout()
        return fig

    def _plot_task_space_vs_ref(self, logs, which="L"):
        """Task-space vs reference: pos, vel, rotvec, ang vel, plus rot err norm. Returns fig."""
        if not logs:
            return None
        t   = np.array([e["t"] for e in logs]) - logs[0]["t"]

        p      = np.stack([e[f"tcp_{which}"]["p"] for e in logs])
        v      = np.stack([e[f"tcp_{which}"]["v"] for e in logs])
        rvec   = np.stack([e[f"tcp_{which}"]["rvec"] for e in logs])
        w      = np.stack([e[f"tcp_{which}"]["w"] for e in logs])

        p_ref  = np.stack([e[f"tcp_{which}"]["p_ref"] for e in logs])
        v_ref  = np.stack([e[f"tcp_{which}"]["v_ref"] for e in logs])
        rref   = np.stack([e[f"tcp_{which}"]["rvec_ref"] for e in logs])
        w_ref  = np.stack([e[f"tcp_{which}"]["w_ref"] for e in logs])

        fig, axs = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

        for k in range(3):
            axs[0].plot(t, p[:,k]);     axs[0].plot(t, p_ref[:,k], "--")
            axs[1].plot(t, v[:,k]);     axs[1].plot(t, v_ref[:,k], "--")
        axs[0].set_ylabel("pos [m]");   axs[0].legend(["meas","ref"], loc="best")
        axs[1].set_ylabel("vel [m/s]"); axs[1].legend(["meas","ref"], loc="best")

        for k in range(3):
            axs[2].plot(t, rvec[:,k]);  axs[2].plot(t, rref[:,k], "--")
            axs[3].plot(t, w[:,k]);     axs[3].plot(t, w_ref[:,k], "--")
        axs[2].set_ylabel("rotvec [rad]"); axs[2].legend(["meas","ref"], loc="best")
        axs[3].set_ylabel("ω [rad/s]");    axs[3].legend(["meas","ref"], loc="best")

        err_norm = np.linalg.norm(rref - rvec, axis=1)
        axs[4].plot(t, err_norm)
        axs[4].set_ylabel("‖rot err‖ [rad]")
        axs[4].set_xlabel("t [s]")

        for ax in axs: ax.grid(True)
        fig.suptitle(f"Task-space vs Ref ({which})")
        fig.tight_layout()
        return fig

    def plot(self):
        """
        Save 4 figures:
        L_joint, L_task, R_joint, R_task
        Filenames follow your pattern and go into ./plots.
        """
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not self.control_data:
            print("No logs to plot.")
            return

        os.makedirs("plots", exist_ok=True)
        ts = current_datetime or datetime.now().strftime("%Y%m%d_%H%M%S")
        rid_L = getattr(self.robotL, "robot_id", "L")
        rid_R = getattr(self.robotR, "robot_id", "R")

        # LEFT
        figJL = self._plot_joint_space(self.control_data, "L")
        if figJL:
            fJL = f"plots/pid_force_pose_ff_terms_{ts}_{rid_L}_joint.png"
            plt.savefig(fJL, dpi=150, bbox_inches="tight"); plt.close(figJL)
            print(f"PID plot saved: {fJL}")

        figTL = self._plot_task_space_vs_ref(self.control_data, "L")
        if figTL:
            fTL = f"plots/pid_force_pose_ff_terms_{ts}_{rid_L}_task.png"
            plt.savefig(fTL, dpi=150, bbox_inches="tight"); plt.close(figTL)
            print(f"PID plot saved: {fTL}")

        # RIGHT
        figJR = self._plot_joint_space(self.control_data, "R")
        if figJR:
            fJR = f"plots/pid_force_pose_ff_terms_{ts}_{rid_R}_joint.png"
            plt.savefig(fJR, dpi=150, bbox_inches="tight"); plt.close(figJR)
            print(f"PID plot saved: {fJR}")

        figTR = self._plot_task_space_vs_ref(self.control_data, "R")
        if figTR:
            fTR = f"plots/pid_force_pose_ff_terms_{ts}_{rid_R}_task.png"
            plt.savefig(fTR, dpi=150, bbox_inches="tight"); plt.close(figTR)
            print(f"PID plot saved: {fTR}")

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
    
    def get_qddot(self):
        a = np.array(self.rtde_receive.getActualQD(), dtype=float)  # joint accelerations
        return a

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
    
    def get_Lambda(self, J, M):
        return np.linalg.inv(J @ np.linalg.inv(M) @ J.T) 

    def get_D(self, K, Lambda):
        D = sqrtm(Lambda) * sqrtm(K) + sqrtm(K) * sqrtm(Lambda)
        return D
    
    def get_wrench_desired(self, D, K, p, v, R, w, p_ref, v_ref, R_ref, w_ref):
        # velocity error
        V_ref = np.hstack((v_ref, w_ref))
        V = np.hstack((v, w))
        Xd_err = V_ref - V

        # position error
        P_err = p_ref - p
        # rotation error
        R_err = R.T @ R_ref #try log later
        r_err = RR.from_matrix(R_err).as_rotvec() 

        X_err = np.hstack((P_err, r_err))

        wrench = D @ Xd_err + K @ X_err

        return wrench