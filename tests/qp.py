import numpy as np
from control.impedance_qp_controller import URImpedanceController, JointOptimization

if __name__ == "__main__":
    K = np.diag([3000, 3000, 3000, 300, 300, 300])

    robotL = URImpedanceController(
        "192.168.1.33", K=K
    )
    robotR = URImpedanceController(
        "192.168.1.66", K=K
    )

    # reference
    trajectory = "motion_planner/trajectories/lifting.npz"
    Hz = 100

    optimizer = JointOptimization(robotL, robotR, Hz, trajectory)

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

        robotR.go_to_approach()
        robotL.go_to_approach()

        robotR.wait_for_commands()
        robotL.wait_for_commands()
        robotR.wait_until_done()
        robotL.wait_until_done()


        robotR.wait_for_control()
        robotL.wait_for_control()

        robotR.plot_data3D()
        robotL.plot_data3D()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robotR.stop_control()
        robotL.stop_control()
    except Exception as e:
        print(f"An error occurred: {e}")
        robotR.stop_control()
        robotL.stop_control()
    finally:
        robotR.disconnect()
        robotL.disconnect()
        print("Robot disconnected")
