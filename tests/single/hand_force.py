from control.pid_controller import URForceController
import numpy as np
from numpy import pi


if __name__ == "__main__":
    hz = 100

    reference_force = 12.5
    base_force = 12.5
    factor = base_force / reference_force

    kp_f = 0.001 * factor
    ki_f = 0.0000 * factor
    kd_f = 0.0002 * factor

    kp_p = 0
    ki_p = 0.0
    kd_p = 0.0

    kp_r = 0  # 0.8
    ki_r = 0
    kd_r = 0  # 0.05

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
        kp_r=kp_r,
        ki_r=ki_r,
        kd_r=kd_r,
    )
    robot.alpha = alpha

    try:
        robot.go_home()
        robot.wait_for_commands()

        robot.wait_until_done()

        robot.control_to_target_manual(
            reference_position=None,
            reference_rotation=None,
            reference_force=reference_force,
            direction=[0, 0, -1],
            distance_cap=0.4,
            timeout=15.0,
        )
        robot.wait_for_control()

        robot.plot_data3D()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robot.stop_control()
    finally:
        robot.disconnect()
        print("Robot disconnected")
