from control.PID.pid_ff_controller import URForceController
import numpy as np

if __name__ == "__main__":
    robot = URForceController("192.168.1.33")

    # robot.go_to_approach()
    # robot.wait_for_commands()
    # robot.wait_until_done()

    state = robot.get_state()

    print(state["joints"])

    # #pose_world = state["pose_world"][:3]
    # pose_world = state["gripper_world"][:3]

    # grasp_data = robot.get_grasping_data_both()
    # grasping_point_world = robot.get_grasping_data()[0]
    # approach_point_world = robot.get_grasping_data()[1]
    # normal_world = robot.get_grasping_data()[2]

    # #pose_base = state["pose"][:3]
    # pose_base = state["gripper_base"][:3]

    # grasping_point_base = robot.world_point_2_robot(grasping_point_world)
    # approach_point_base = robot.world_point_2_robot(approach_point_world)
    # normal_base = robot.world_vector_2_robot(normal_world)


    # d_L_euckl_world = np.linalg.norm(pose_world - grasping_point_world)
    # d_L_normal_world = float((pose_world - grasping_point_world) @ normal_world)
    # d_L_euckl_base = np.linalg.norm(pose_base - grasping_point_base)
    # d_L_normal_base = float((pose_base - grasping_point_base) @ normal_base)

    # print("=== Robot State World ===")
    # print("Pose world:", pose_world)
    # print("Approach point world:", approach_point_world)
    # print("Grasping point world:", grasping_point_world)
    # print("Normal world:", normal_world)
    # print("\n")
    # print("=== Robot State Base ===")
    # print("Pose base:", pose_base)
    # print("Approach point base:", approach_point_base)
    # print("Grasping point base:", grasping_point_base)
    # print("Normal base:", normal_base)
    # print("\n")
    # print("=== Distances ===")
    # print("Euclidean world distance:", d_L_euckl_world)
    # print("Normal world distance:", d_L_normal_world)
    # print("Euclidean base distance:", d_L_euckl_base)
    # print("Normal base distance:", d_L_normal_base)
