from control.ur_force_controller import URForceController
import numpy as np

if __name__ == "__main__":
    hz = 50

    kp_f = 0.0
    ki_f = 0.0
    kd_f = 0.0

    kp_p = 1
    ki_p = 0.0
    kd_p = 0.0

    alpha = 0.05

    robot = URForceController(
        "192.168.1.66",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
    )
    robot.alpha = alpha

    offset = np.array([0.0, 0.0, -0.15])  # downwards
    pose = np.array(robot.get_state()["pose"][:3])  # Get the current pose (x, y, z)
    ref_pose = pose + offset
    print(ref_pose)

    try:
        robot.control_to_target_manual(
            target_position=ref_pose,
            reference_force=10.0,
            direction=[0, 0, -1],
            distance_cap=0.3,
            timeout=15.0,
        )
        robot.wait_for_control()

        robot.plot_data()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot.disconnect()
        print("Robot disconnected")
