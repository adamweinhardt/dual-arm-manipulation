from control.PID.pid_ff_controller import URForceController
import numpy as np

if __name__ == "__main__":
    robot = URForceController("192.168.1.33")

    # robot.go_to_approach()
    # state = robot.get_state()
    # grasping_point = robot.get_grasping_data()[0]
    # print("Pose world:", state["gripper_world"][:3])
    # print("Pose base:", state["gripper_base"][:3])
    grasping_point = np.array([ 0.11190338 ,-0.01794107,  0.08563561])
    print("Grasping point world:", grasping_point)
    print("Grasping point base:", robot.world_point_2_robot(grasping_point))
