import numpy as np
from control.impedance_qp_right import RightArmOptimization, URImpedanceController

if __name__ == "__main__":
    # Impedance gains (same pattern you used)
    K = np.diag([3000, 3000, 3000, 200, 200, 200])

    # ---- RIGHT robot only ----
    robotR = URImpedanceController(
        "192.168.1.66", K=K
    )

    # Trajectory (BOX frame); adjust path if needed
    trajectory = "motion_planner/trajectories/lifting.npz"
    Hz = 60

    # Create right-arm optimizer; start with rotation disabled to simplify debugging
    optimizer = RightArmOptimization(
        robotR=robotR,
        Hz=Hz,
        trajectory_npz_path=trajectory,
        R_world_box0=None,        # or pass a 3x3 if you want a specific BOX→WORLD alignment
        disable_rotation=True,    # set False later when linear part looks good
    )

    try:
        # --- Bring the right arm to a known state (adapt these to your setup) ---
        robotR.go_home()
        robotR.wait_for_commands()
        robotR.wait_until_done()

        robotR.go_to_approach()
        robotR.wait_for_commands()
        robotR.wait_until_done()

        # If your pipeline needs it, you can inspect grasp data;
        # RightArmOptimization doesn't require other-robot data.
        # pR, _, _ = robotR.get_grasping_data()

        # --- Run the single-arm QP control loop ---
        optimizer.run()

        # --- Plots (RIGHT only) ---
        optimizer.plot_taskspace_tracking(title_prefix="TaskspaceTracking_R")
        optimizer.plot_qp_and_jointspace(title_prefix="QP_and_Jointspace_R")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robotR.stop_control()
    except Exception as e:
        print(f"An error occurred: {e}")
        robotR.stop_control()
    finally:
        robotR.disconnect()
        print("Robot disconnected")
