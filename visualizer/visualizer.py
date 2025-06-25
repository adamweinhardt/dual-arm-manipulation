import zmq
import time
import json
import numpy as np

import robotic as ry

from robot_ipc_control.pose_estimation.transform_utils import (
    rotation_matrix_to_quaternion,
)
from robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator
from robot_ipc_control.pose_estimation.scene_utils import make_scene, get_robot_joints
from robot_ipc_control.examples.robot_interface import urtde_to_rai


class Visualizer:
    def __init__(self):
        config_path = "robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
        with open(config_path, "r") as f:
            self.scene_config = json.load(f)

        # create scene
        self.C, self.box_names, self.robot_names = make_scene(self.scene_config)

        # get robot joint names
        self.robot_joint_names = {}
        for robot_name in self.robot_names:
            self.robot_joint_names[robot_name] = get_robot_joints(self.C, robot_name)

        # ZMQ setup for robots
        self.robot_context = zmq.Context()
        self.robot_sockets = {}

        robot_ports = [5556, 5559]

        for i, port in enumerate(robot_ports):
            socket = self.robot_context.socket(zmq.SUB)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.connect(f"tcp://127.0.0.1:{port}")
            self.robot_sockets[f"robot_{i}"] = socket

        # box tracking
        box_port = self.scene_config.get("port", 5557)
        self.box_estimator = BoardPoseEstimator(f"tcp://localhost:{box_port}")
        self.box_estimator.start()

        # grasping points subscription
        self.grasping_context = zmq.Context()
        self.grasping_socket = self.grasping_context.socket(zmq.SUB)
        self.grasping_socket.setsockopt(zmq.CONFLATE, 1)
        self.grasping_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        grasping_port = 5560
        self.grasping_socket.connect(f"tcp://127.0.0.1:{grasping_port}")

        self.current_grasping_data = {}

        self.marker_frames = {}

        print(f"Grasping points subscriber connected to port {grasping_port}")

        self.running = False
        print(
            f"Visualizer ready: {len(self.robot_names)} robots, {len(self.box_names)} boxes"
        )
        print("Grasping points visualization enabled on port 5560")
        print("Available box names:", self.box_names)
        print("Available robot names:", self.robot_names)

    def update_robots(self):
        """Update robot poses from ZMQ streams"""
        for socket_id, socket in self.robot_sockets.items():
            try:
                state_data = socket.recv_json(flags=zmq.NOBLOCK)

                if "Q" in state_data and "robot_id" in state_data:
                    if not state_data["Q"]:
                        continue

                    robot_id = state_data["robot_id"]
                    robot_index = int(robot_id.split("_")[1])

                    if robot_index < len(self.robot_names):
                        robot_name = self.robot_names[robot_index]

                        try:
                            joint_positions = urtde_to_rai(state_data["Q"])
                            joint_names = self.robot_joint_names[robot_name]

                            self.C.setJointState(joint_positions, joint_names)

                        except Exception as e:
                            print(f"Joint update error for {robot_name}: {e}")

            except zmq.Again:
                continue
            except Exception as e:
                continue

    def update_boxes(self):
        try:
            tracked_ids = self.box_estimator.get_tracked_board_ids()

            for box_id in tracked_ids:
                box_pose = self.box_estimator.get_pose(box_id)
                is_stable = self.box_estimator.is_stable(box_id)

                if int(box_id) < len(self.box_names):
                    box_name = self.box_names[int(box_id)]

                    if is_stable and box_pose is not None:
                        position = np.array(box_pose[:3])
                        quaternion = np.array(box_pose[3:])

                        self.C.getFrame(box_name).setRelativePosition(position)
                        self.C.getFrame(box_name).setRelativeQuaternion(quaternion)
                        self.C.getFrame(box_name).setColor([0.5, 0.5, 0.5, 0.8])
                    else:
                        self.C.getFrame(box_name).setColor([0.3, 0.3, 0.3, 0.5])

        except Exception:
            pass

    def update_grasping_points(self):
        try:
            message = self.grasping_socket.recv_json(flags=zmq.NOBLOCK)

            if "grasping_points" in message:
                self.current_grasping_data = message["grasping_points"]
                # print(f"Received grasping data for {len(self.current_grasping_data)} boxes")

                for box_id, grasp_data in self.current_grasping_data.items():
                    self._update_grasping_markers(box_id, grasp_data)

        except zmq.Again:
            pass
        except Exception as e:
            print(f"Grasping points update error: {e}")

    def _update_grasping_markers(self, box_id: str, grasp_data: dict):
        try:
            marker1_name = f"grasp_point1_box_{box_id}"
            marker2_name = f"grasp_point2_box_{box_id}"

            point1 = np.array(grasp_data["point1"])
            point2 = np.array(grasp_data["point2"])

            # print(f"Creating markers at: {point1} and {point2}")

            if marker1_name not in self.marker_frames:
                print(f"Creating new marker: {marker1_name}")
                marker1 = self.C.addFrame(marker1_name)
                marker1.setShape(ry.ST.sphere, [0.02])
                marker1.setColor([1.0, 0.0, 0.0, 0.9])
                self.marker_frames[marker1_name] = marker1
            else:
                marker1 = self.marker_frames[marker1_name]
                # print(f"Updating existing marker: {marker1_name}")

            print("Point1")
            print(point1)

            marker1.setPosition(point1)

            if marker2_name not in self.marker_frames:
                # print(f"Creating new marker: {marker2_name}")
                marker2 = self.C.addFrame(marker2_name)
                marker2.setShape(ry.ST.sphere, [0.02])
                marker2.setColor([1.0, 0.0, 0.0, 0.9])
                self.marker_frames[marker2_name] = marker2
            else:
                marker2 = self.marker_frames[marker2_name]
                # print(f"Updating existing marker: {marker2_name}")

            print("Point2")
            print(point2)
            marker2.setPosition(point2)

        except Exception as e:
            print(f"Marker update error for box {box_id}: {e}")
            import traceback

            traceback.print_exc()

    def start(self):
        self.running = True
        try:
            while self.running:
                self.update_robots()
                self.update_boxes()
                self.update_grasping_points()
                self.C.view(False)
                time.sleep(0.05)  # 20Hz

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean exit with grasping socket cleanup"""
        self.running = False

        for socket in self.robot_sockets.values():
            socket.close()
        self.robot_context.term()

        self.grasping_socket.close()
        self.grasping_context.term()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.start()
