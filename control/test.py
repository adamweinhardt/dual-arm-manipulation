from control.ur_force_controller import URForceController
import numpy as np

if __name__ == "__main__":
    hz = 50

    kp_f = 0.005
    ki_f = 0.000
    kd_f = 0.0001

    kp_p = 0.5
    ki_p = 0.2
    kd_p = 0.002

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
    pose = np.array(robot.get_state()["joints"])
    print(pose)
