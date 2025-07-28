from control.ur_pid_controller import URForceController
import numpy as np

if __name__ == "__main__":
    hz = 100

    kp_f = 0.0
    ki_f = 0.0
    kd_f = 0.0

    kp_p = 0
    ki_p = 0.0
    kd_p = 0.0

    kp_r = 0.8
    ki_r = 0
    kd_r = 0.05

    alpha = 0.99

    robot = URForceController(
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
    )
    robot.alpha = alpha

    test = [
        {"time": 2.0, "position": [0, 0, 0], "rotation": [0, 0, np.pi / 4]},
    ]

    try:
        robot.go_home()

        robot.wait_for_commands()

        robot.wait_until_done()

        robot.control_to_target_manual(
            reference_position=None,
            reference_rotation=None,
            reference_force=10.0,
            direction=[0, 0, -1],
            distance_cap=0.3,
            timeout=10.0,
            pose_updates=test,
        )
        robot.wait_for_control()

        robot.plot_data3D()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robot.stop_control()
    finally:
        robot.disconnect()
        print("Robot disconnected")
