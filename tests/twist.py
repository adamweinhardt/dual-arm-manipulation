from control.pid_controller import URForceController
import numpy as np
import time


if __name__ == "__main__":
    hz = 100
    reference_force = 50 # 150
    base_force = 12.5
    factor = base_force / reference_force

    kp_f = 0.0022 * factor
    ki_f = 0.0000 * factor
    kd_f = 0.0001 * factor

    kp_p = 2  # 0.5
    ki_p = 0.00005
    kd_p = 0.25  # 0.0025

    kp_r = 1.8
    ki_r = 0
    kd_r = 0.2

    kp_t=0.01,
    ki_t=0, 
    kd_t=0.001

    alpha = 0.85
    deadzone_threshold = None
    trajectory = "motion_planner/trajectories/pick_and_twist.npz"

    robotL = URForceController(
        "192.168.1.33",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
        kp_r=kp_r,
        ki_r=ki_r,
        kd_r=kd_r,
        kp_t=kp_t,
        ki_t=ki_t,
        kd_t=kd_t,
    )
    robotR = URForceController(
        "192.168.1.66",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
        kp_r=kp_r,
        ki_r=ki_r,
        kd_r=kd_r,
        kp_t=kp_t,
        ki_t=ki_t,
        kd_t=kd_t,
    )

    robotL.alpha = alpha
    robotR.alpha = alpha

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

        pL, _, _ = robotL.get_grasping_data()
        pR, _, _ = robotR.get_grasping_data()
        robotL.other_robot_grasp_point = np.array(pR)
        robotR.other_robot_grasp_point = np.array(pL)

        time.sleep(0.1)

        robotR.control_to_target(
            reference_force=reference_force,
            distance_cap=0.5,
            timeout=15,
            trajectory=trajectory,
            deadzone_threshold=deadzone_threshold,
        )

        robotL.control_to_target(
            reference_force=reference_force,
            distance_cap=0.5,
            timeout=15,
            trajectory=trajectory,
            deadzone_threshold=deadzone_threshold,
        )

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
