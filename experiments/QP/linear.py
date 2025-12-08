from control.QP.qp_full import URImpedanceController, DualArmImpedanceAdmittanceQP
from utils.utils import diag6
import numpy as np
import time



if __name__ == "__main__":
    K = np.diag([15, 15, 15, 0.2, 0.2, 0.2])
    
    robot_L = URImpedanceController("192.168.1.33", K=K)
    robot_R = URImpedanceController("192.168.1.66", K=K)

    # Weights
    W_imp = diag6([1.0, 1.0, 1.0, 1e3, 1e3, 1e3])
    #W_imp = diag6([0,0,0,0,0,0])
    W_grasp = diag6([5e1, 5e1, 5e1, 2e5, 2e5, 2e5])
    #W_grasp = diag6([0, 0, 0, 0, 0, 0])
    lambda_reg = 1e-6

    M_a = 27.0
    K_a = 400.0
    D_a = 2400.0  # or 2*sqrt(M_a*K_a)
    v_max = 0.05
    Fn_ref = 35.0


    date = time.strftime("%Y%m%d-%H%M%S")
    version = "QP" # PID, PID_dz, PID_ff, QP
    box = "bw" # migros, vention
    traj = "angular"
    traj_path = f"motion_planner/trajectories/{traj}.npz"
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

        ctrl._init_grasping_data()
        
        print("Moving to Approach...")
        robot_L.go_to_approach(); robot_R.go_to_approach()
        robot_L.wait_for_commands(); robot_R.wait_for_commands()
        robot_L.wait_until_done(); robot_R.wait_until_done()

        # run controller
        ctrl.run()

        ctrl.save_everything(f"experiments/QP/logs/{traj}_{version}_{box}_{date}")

    except KeyboardInterrupt:
        print("Interrupted")
        robot_R.stop_control()
        robot_L.stop_control()
    finally:
        try:
            robot_L.disconnect(); robot_R.disconnect()
        except: pass
        print("Robots disconnected.")
