from control.ur_force_controller import URForceController

if __name__ == "__main__":
    hz = 50
    kp = 0.01
    ki = 0.0
    kd = 0.0

    robot = URForceController("192.168.1.66", hz=hz, kp=kp, ki=ki, kd=kd)

    try:
        print("\nStarting force control...")
        robot.force_control_to_target(
            reference_force=7.5,
            direction=[0, 0, -1],
            distance_cap=0.3,
            timeout=20.0,
        )
        robot.wait_for_force_control()

        print("Force control complete!")
        print("Final force reading:", robot.get_current_force())

        robot.plot_force_data()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot.disconnect()
        print("Robot disconnected")
