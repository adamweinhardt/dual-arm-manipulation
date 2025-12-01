import zmq
import time
import json
import numpy as np
import threading
import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from robot_ipc_control.pose_estimation.transform_utils import quat_to_rotmat

from robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator


@dataclass
class Box:
    id: int
    position: np.ndarray  # [x, y, z]
    rotation_matrix: np.ndarray  # 3x3
    quaternions: np.ndarray  # [w, x, y, z]
    confidence: float
    x_dim: float
    y_dim: float
    z_dim: float


@dataclass
class FacePair:
    face0_center: np.ndarray
    face1_center: np.ndarray
    face0_normal: np.ndarray
    face1_normal: np.ndarray
    pair_type: str


@dataclass
class GraspingPair:
    box_id: str
    grasping_point0: np.ndarray
    grasping_point1: np.ndarray
    normal0: np.ndarray
    normal1: np.ndarray
    approach_point0: np.ndarray
    approach_point1: np.ndarray
    pair_type: str
    confidence: float
    total_distance: float
    robot_assignment: Dict[str, str]


def get_dimensions_from_config(board_config_path: str) -> tuple:
    """
    Get box dimensions (x_span, y_span, z_span) from board config,
    corresponding to the ArUco board's local X, Y, Z axes.
    """
    with open(board_config_path, "r") as f:
        board_data = json.load(f)

    min_pt = np.array([float("inf")] * 3)
    max_pt = np.array([float("-inf")] * 3)

    for marker in board_data["markers"]:
        for corner in marker["corners"]:
            for k in range(3):
                min_pt[k] = min(min_pt[k], corner[k])
                max_pt[k] = max(max_pt[k], corner[k])

    dims = max_pt - min_pt
    return dims[0], dims[1], dims[2]


class GraspingPointsCalculator:
    """Calculates optimal grasping points using BoardPoseEstimator"""

    def __init__(self, config_path: str, approach_offset: float = 0.15):
        self.config_path = config_path
        self.approach_offset = approach_offset
        self.robot_poses = self._load_robot_poses()
        self.box_dimensions_aruco_axes = self._load_box_dimensions()

        with open(config_path, "r") as f:
            config = json.load(f)

        port = config.get("port", 5557)
        self.pose_estimator = BoardPoseEstimator(f"tcp://localhost:{port}")
        self.pose_estimator.start()

    def _load_robot_poses(self) -> List[np.ndarray]:
        """Load robot poses from config file"""
        with open(self.config_path, "r") as f:
            config = json.load(f)

        robot_poses = []
        for robot_pose_file in config["robots"]:
            pose_matrix = np.load(robot_pose_file)
            robot_poses.append(pose_matrix)

        return robot_poses

    def _load_box_dimensions(self) -> Dict[int, tuple]:
        """Load box dimensions (x_dim, y_dim, z_dim) from config,
        aligned with ArUco board's local axes."""
        with open(self.config_path, "r") as f:
            config = json.load(f)

        config_dir = os.path.dirname(os.path.dirname(self.config_path))
        box_dimensions_aruco_axes = {}

        for i, board_config in enumerate(config["boards"]):
            if not os.path.isabs(board_config):
                board_config = os.path.join(config_dir, board_config.lstrip("./"))

            x_dim, y_dim, z_dim = get_dimensions_from_config(board_config)
            box_dimensions_aruco_axes[i] = (x_dim, y_dim, z_dim)

        return box_dimensions_aruco_axes

    def get_current_boxes(self) -> List[Box]:
        """Get current boxes from the pose estimator"""
        tracked_ids = self.pose_estimator.get_tracked_board_ids()
        boxes = []

        for board_id in tracked_ids:
            board_id_int = int(board_id)

            if board_id_int in self.box_dimensions_aruco_axes:
                pose = self.pose_estimator.get_pose(board_id)
                confidence = self.pose_estimator.get_confidence(board_id)
                is_stable = self.pose_estimator.is_stable(board_id)

                if pose is not None and is_stable:
                    estimated_position = pose[:3]
                    quaternion = pose[3:7]
                    rotation_matrix = quat_to_rotmat(quaternion)

                    height, width, depth = self.box_dimensions_aruco_axes[board_id_int]

                    box_position = estimated_position

                    box = Box(
                        id=board_id_int,
                        position=box_position,
                        rotation_matrix=rotation_matrix,
                        quaternions=quaternion,
                        confidence=confidence,
                        x_dim=height,
                        y_dim=width,
                        z_dim=depth,
                    )
                    boxes.append(box)

        return boxes

    def get_box_face_centers(self, box: Box) -> Dict[str, np.ndarray]:
        """
        Get face centers relative to the box's center, in world coordinates.
        This calculation matches how ry.ST.box uses the dimensions and rotation.
        """
        x_dim, y_dim, z_dim = box.x_dim, box.y_dim, box.z_dim

        half_x_dim, half_y_dim, half_z_dim = x_dim / 2, y_dim / 2, z_dim / 2

        local_x_axis_world = box.rotation_matrix[:, 0]
        local_y_axis_world = box.rotation_matrix[:, 1]
        local_z_axis_world = box.rotation_matrix[:, 2]

        faces_world = {
            "front": box.position + local_x_axis_world * half_x_dim,  # +X face
            "back": box.position - local_x_axis_world * half_x_dim,  # -X face
            "right": box.position + local_y_axis_world * half_y_dim,  # +Y face
            "left": box.position - local_y_axis_world * half_y_dim,  # -Y face
            "top": box.position + local_z_axis_world * half_z_dim,  # +Z face
            "bottom": box.position - local_z_axis_world * half_z_dim,  # -Z face
        }

        return faces_world

    def get_face_normals(self, box: Box) -> Dict[str, np.ndarray]:
        """
        Get face normal vectors in world coordinates, aligned with ArUco board's local axes.
        """
        local_x_axis_world = box.rotation_matrix[:, 0]
        local_y_axis_world = box.rotation_matrix[:, 1]
        local_z_axis_world = box.rotation_matrix[:, 2]

        normals_world = {
            "front": local_x_axis_world,  # +X direction (outward from 'front' face)
            "back": -local_x_axis_world,  # -X direction (outward from 'back' face)
            "right": local_y_axis_world,  # +Y direction (outward from 'right' face)
            "left": -local_y_axis_world,  # -Y direction (outward from 'left' face)
            "top": local_z_axis_world,  # +Z direction (outward from 'top' face)
            "bottom": -local_z_axis_world,  # -Z direction (outward from 'bottom' face)
        }

        return normals_world

    def get_all_face_pairs(self, box: Box) -> List[FacePair]:
        """Get all opposing face pairs for grasping"""
        face_centers = self.get_box_face_centers(box)
        face_normals = self.get_face_normals(box)

        pairs = []

        opposing_pairs = [
            ("front", "back", "front_back_X_axis"),
            ("left", "right", "left_right_Y_axis"),
            ("top", "bottom", "top_bottom_Z_axis"),
        ]

        for face0_name, face1_name, pair_type in opposing_pairs:
            pairs.append(
                FacePair(
                    face0_center=face_centers[face0_name],
                    face1_center=face_centers[face1_name],
                    face0_normal=face_normals[face0_name],
                    face1_normal=face_normals[face1_name],
                    pair_type=pair_type,
                )
            )

        return pairs

    def get_approach_points(
        self,
        grasp_point0: np.ndarray,
        grasp_point1: np.ndarray,
        normal0: np.ndarray,
        normal1: np.ndarray,
        offset: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if offset is None:
            offset = self.approach_offset

        normal0_unit = normal0 / np.linalg.norm(normal0)
        normal1_unit = normal1 / np.linalg.norm(normal1)

        approach_point0 = grasp_point0 + normal0_unit * offset
        approach_point1 = grasp_point1 + normal1_unit * offset

        return approach_point0, approach_point1

    def calculate_face_robot_distances(self, face_center: np.ndarray) -> List[float]:
        """Calculate distances from a face center to all robot bases"""
        distances = []
        for i, robot_pose in enumerate(self.robot_poses):
            robot_position = robot_pose[:3, 3]
            distance = np.linalg.norm(face_center - robot_position)
            distances.append(distance)

        return distances

    def find_best_grasping_pair(self, box: Box) -> Optional[GraspingPair]:
        """Find the best opposing face pair for dual-arm grasping with consistent robot assignment"""
        all_pairs = self.get_all_face_pairs(box)

        if not all_pairs:
            return None

        best_pair = None
        best_score = float("inf")
        best_analysis = None

        for i, pair in enumerate(all_pairs):
            face0_distances = self.calculate_face_robot_distances(pair.face0_center)
            face1_distances = self.calculate_face_robot_distances(pair.face1_center)

            assignment0_score = face0_distances[0] + face1_distances[1]
            assignment1_score = face1_distances[0] + face0_distances[1]

            if assignment0_score < assignment1_score:
                total_distance = assignment0_score
                robot_assignment = {
                    "0": pair.pair_type.split("_")[0],
                    "1": pair.pair_type.split("_")[1],
                }
                # Robot 0 gets face0, Robot 1 gets face1
                robot0_face_center = pair.face0_center
                robot0_normal = pair.face0_normal
                robot1_face_center = pair.face1_center
                robot1_normal = pair.face1_normal
            else:
                total_distance = assignment1_score
                robot_assignment = {
                    "0": pair.pair_type.split("_")[1],
                    "1": pair.pair_type.split("_")[0],
                }
                # Robot 0 gets face1, Robot 1 gets face0
                robot0_face_center = pair.face1_center
                robot0_normal = pair.face1_normal
                robot1_face_center = pair.face0_center
                robot1_normal = pair.face0_normal

            if total_distance < best_score:
                best_score = total_distance
                best_pair = pair
                best_analysis = {
                    "total_distance": total_distance,
                    "robot_assignment": robot_assignment,
                    "robot0_face_center": robot0_face_center,
                    "robot0_normal": robot0_normal,
                    "robot1_face_center": robot1_face_center,
                    "robot1_normal": robot1_normal,
                }

        if best_pair is None:
            return None

        # Use the consistently assigned robot positions
        robot0_grasp_point = best_analysis["robot0_face_center"]
        robot0_normal = best_analysis["robot0_normal"]
        robot1_grasp_point = best_analysis["robot1_face_center"]
        robot1_normal = best_analysis["robot1_normal"]

        # Calculate approach points with consistent assignment
        approach_point0, approach_point1 = self.get_approach_points(
            robot0_grasp_point,  # Robot 0's grasp point
            robot1_grasp_point,  # Robot 1's grasp point
            robot0_normal,  # Robot 0's normal
            robot1_normal,  # Robot 1's normal
        )

        return GraspingPair(
            box_id=str(box.id),
            grasping_point0=robot0_grasp_point,  # Always robot 0
            grasping_point1=robot1_grasp_point,  # Always robot 1
            normal0=robot0_normal,  # Always robot 0
            normal1=robot1_normal,  # Always robot 1
            approach_point0=approach_point0,  # Always robot 0
            approach_point1=approach_point1,  # Always robot 1
            pair_type=best_pair.pair_type,
            confidence=float(box.confidence),
            total_distance=float(best_analysis["total_distance"]),
            robot_assignment=best_analysis["robot_assignment"],
        )

    def disconnect(self):
        """Clean shutdown"""
        self.pose_estimator.stop()


class GraspingPointsPublisher:
    """Publishes optimal grasping points via ZMQ"""

    def __init__(
        self,
        config_path: str,
        port: int = 5560,
        publish_rate_hz: float = 20.0,
        approach_offset: float = 0.05,
    ):
        self.config_path = config_path
        self.port = port
        self.publish_rate_hz = publish_rate_hz

        self.calculator = GraspingPointsCalculator(config_path, approach_offset)

        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{port}")

        self.publishing = False
        self.publisher_thread = None

        print(f"Grasping points publisher ready on port {port}")
        print(f"Approach offset: {approach_offset}m")

    def start_publishing(self):
        """Start the publishing thread"""
        if self.publishing:
            return

        self.publishing = True
        self.publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher_thread.start()
        print("Grasping points publishing started")

    def stop_publishing(self):
        """Stop the publishing thread"""
        if self.publishing:
            self.publishing = False
            if self.publisher_thread:
                self.publisher_thread.join()
            print("Grasping points publishing stopped")

    def _publish_loop(self):
        """Main publishing loop with explicit robot assignment"""
        interval = 1.0 / self.publish_rate_hz

        while self.publishing:
            try:
                start_time = time.time()

                boxes = self.calculator.get_current_boxes()

                if boxes:
                    grasping_data = {}

                    for box in boxes:
                        grasping_pair = self.calculator.find_best_grasping_pair(box)

                        if grasping_pair:
                            grasping_data[grasping_pair.box_id] = {
                                "grasping_point0": grasping_pair.grasping_point0.tolist(),  # always robot 0
                                "grasping_point1": grasping_pair.grasping_point1.tolist(),  # always robot 1
                                "normal0": grasping_pair.normal0.tolist(),  # always robot 0
                                "normal1": grasping_pair.normal1.tolist(),  # always robot 1
                                "approach_point0": grasping_pair.approach_point0.tolist(),  # always robot 0
                                "approach_point1": grasping_pair.approach_point1.tolist(),  # always robot 1
                                "approach_offset": self.calculator.approach_offset,
                                "pair_type": grasping_pair.pair_type,
                                "confidence": grasping_pair.confidence,
                                "total_distance": grasping_pair.total_distance,
                                "robot_assignment": grasping_pair.robot_assignment,
                                "timestamp": float(time.time()),
                                "box_x_dim": box.x_dim,
                                "box_y_dim": box.y_dim,
                                "box_z_dim": box.z_dim,
                            }
                        print("-----")
                        print(box.x_dim)
                        print(box.y_dim)
                        print(box.z_dim)
                        print("-----")
                    if grasping_data:
                        message = {
                            "timestamp": float(time.time()),
                            "grasping_points": grasping_data,
                        }
                        self.publisher.send_json(message)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Publishing error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)

    def disconnect(self):
        """Clean shutdown"""
        self.stop_publishing()
        self.calculator.disconnect()
        self.publisher.close()
        self.context.term()


if __name__ == "__main__":
    config_path = (
        "robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
    )

    publisher = GraspingPointsPublisher(config_path, approach_offset=0.025)

    try:
        publisher.start_publishing()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        publisher.disconnect()
