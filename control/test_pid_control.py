from control.ur_pid_controller import URForceController
import numpy as np

if __name__ == "__main__":
    hz = 100

    kp_f = 0.005
    ki_f = 0.0
    kd_f = 0.0

    kp_p = 0.0
    ki_p = 0.0
    kd_p = 0.0

    alpha = 0.99

    robot = URForceController(
        "192.168.1.33",
        hz=hz,
        kp_f=kp_f,
        ki_f=ki_f,
        kd_f=kd_f,
        kp_p=kp_p,
        ki_p=ki_p,
        kd_p=kd_p,
    )
    robot.alpha = alpha

    offset = np.array([0.0, 0.0, -0.10])  # downwards
    pose = np.array(
        robot.get_state()["pose_world"][:3]
    )  # Get the current pose (x, y, z)
    ref_pose = pose + offset

    try:
        robot.control_to_target_manual(
            target_position=ref_pose,
            reference_force=10.0,
            direction=[0, 0, -1],
            distance_cap=0.3,
            timeout=10.0,
        )
        robot.wait_for_control()

        robot.plot_data3D()
        robot.plot_forces()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robot.stop_control()
    finally:
        robot.disconnect()
        print("Robot disconnected")
