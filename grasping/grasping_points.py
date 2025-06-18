import numpy as np
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

# Import Box class and BoxTracker from boxes.py
import sys

sys.path.append(os.path.dirname(__file__))
from scripts.boxes import Box, BoxTracker


@dataclass
class FacePair:
    face1_center: np.ndarray
    face2_center: np.ndarray
    face1_normal: np.ndarray
    face2_normal: np.ndarray
    pair_type: str  # "front_back" or "left_right"


def get_robot_poses(config_path: str) -> List[np.ndarray]:
    """Extract robot poses from config file

    Returns:
        List of 4x4 homogeneous transformation matrices for each robot
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get config directory (go up one level from configs/)
    config_dir = os.path.dirname(os.path.dirname(config_path))

    robot_poses = []

    for robot_pose_file in config["robots"]:
        # Resolve path relative to repo root
        if not os.path.isabs(robot_pose_file):
            robot_pose_file = os.path.join(config_dir, robot_pose_file.lstrip("./"))

        # Load the pose matrix
        pose_matrix = np.load(robot_pose_file)
        robot_poses.append(pose_matrix)

    return robot_poses


def get_box_face_centers(box: Box) -> Dict[str, np.ndarray]:
    """Get all 6 face centers in world coordinates"""
    w, h, d = box.width / 2, box.height / 2, box.depth / 2

    # Face centers in box local frame (Z=up, X=forward)
    faces_local = {
        "top": np.array([0, 0, +d]),  # +Z (up)
        "bottom": np.array([0, 0, -d]),  # -Z (down)
        "front": np.array([+w, 0, 0]),  # +X (forward)
        "back": np.array([-w, 0, 0]),  # -X (backward)
        "right": np.array([0, +h, 0]),  # +Y (right)
        "left": np.array([0, -h, 0]),  # -Y (left)
    }

    # Transform to world coordinates
    faces_world = {}
    for name, local_pos in faces_local.items():
        world_pos = box.rotation_matrix @ local_pos + box.position
        faces_world[name] = world_pos

    return faces_world


def get_face_normals(box: Box) -> Dict[str, np.ndarray]:
    """Get face normal vectors in world coordinates"""
    # Face normals in box local frame (Z=up, X=forward)
    normals_local = {
        "top": np.array([0, 0, 1]),  # +Z (up)
        "bottom": np.array([0, 0, -1]),  # -Z (down)
        "front": np.array([1, 0, 0]),  # +X (forward)
        "back": np.array([-1, 0, 0]),  # -X (backward)
        "right": np.array([0, 1, 0]),  # +Y (right)
        "left": np.array([0, -1, 0]),  # -Y (left)
    }

    # Transform to world coordinates (only rotate, don't translate)
    normals_world = {}
    for name, local_normal in normals_local.items():
        world_normal = box.rotation_matrix @ local_normal
        normals_world[name] = world_normal

    return normals_world


def get_all_face_pairs(box: Box) -> List[FacePair]:
    """Get all opposing face pairs (including top/bottom for flipping)"""
    face_centers = get_box_face_centers(box)
    face_normals = get_face_normals(box)

    # Create all opposing pairs - including top/bottom for flipping
    pairs = [
        FacePair(
            face1_center=face_centers["left"],
            face2_center=face_centers["right"],  # Left and Right are opposite
            face1_normal=face_normals["left"],
            face2_normal=face_normals["right"],
            pair_type="left_right",
        ),
        FacePair(
            face1_center=face_centers["front"],
            face2_center=face_centers["back"],  # Front and Back are opposite
            face1_normal=face_normals["front"],
            face2_normal=face_normals["back"],
            pair_type="front_back",
        ),
        FacePair(
            face1_center=face_centers["top"],
            face2_center=face_centers["bottom"],  # Top and Bottom for flipping
            face1_normal=face_normals["top"],
            face2_normal=face_normals["bottom"],
            pair_type="top_bottom",
        ),
    ]

    return pairs


def calculate_face_robot_distances(
    face_center: np.ndarray, robot_poses: List[np.ndarray]
) -> List[float]:
    """Calculate distances from a face center to all robot bases"""
    distances = []
    for robot_pose in robot_poses:
        robot_position = robot_pose[:3, 3]  # Extract position from 4x4 matrix
        distance = np.linalg.norm(face_center - robot_position)
        distances.append(distance)
    return distances


def find_best_grasping_pair(
    box: Box, robot_poses: List[np.ndarray]
) -> Tuple[FacePair, Dict]:
    """Find the best opposing face pair for dual-arm grasping (including flipping)"""
    all_pairs = get_all_face_pairs(box)

    best_pair = None
    best_score = float("inf")
    analysis = {}

    for pair in all_pairs:
        # Calculate distances from each face to each robot
        face1_distances = calculate_face_robot_distances(pair.face1_center, robot_poses)
        face2_distances = calculate_face_robot_distances(pair.face2_center, robot_poses)

        # For good grasping, we want:
        # 1. Robot 0 close to face1, Robot 1 close to face2 (or vice versa)
        # 2. Minimum total distance

        # Try assignment 1: Robot 0 → face1, Robot 1 → face2
        assignment1_score = face1_distances[0] + face2_distances[1]

        # Try assignment 2: Robot 0 → face2, Robot 1 → face1
        assignment2_score = face2_distances[0] + face1_distances[1]

        # Choose better assignment
        if assignment1_score < assignment2_score:
            total_distance = assignment1_score
            robot_assignment = {0: "face1", 1: "face2"}
            distances = {
                "robot0_to_target": face1_distances[0],
                "robot1_to_target": face2_distances[1],
            }
        else:
            total_distance = assignment2_score
            robot_assignment = {0: "face2", 1: "face1"}
            distances = {
                "robot0_to_target": face2_distances[0],
                "robot1_to_target": face1_distances[1],
            }

        analysis[pair.pair_type] = {
            "total_distance": total_distance,
            "robot_assignment": robot_assignment,
            "distances": distances,
            "face1_distances": face1_distances,
            "face2_distances": face2_distances,
        }

        # Update best pair
        if total_distance < best_score:
            best_score = total_distance
            best_pair = pair

    return best_pair, analysis


# Example usage and testing
if __name__ == "__main__":
    # Load robot poses from config
    config_path = "/home/weini/code/robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
    robot_poses = get_robot_poses(config_path)

    # Get real box data from pose estimation
    tracker = BoxTracker(config_path)
    boxes = tracker.get_boxes()

    if not boxes:
        print("No boxes detected! Make sure pose estimation is running.")
        exit()

    # Use first detected box
    box = boxes[0]
    print(f"Using Box {box.id} for analysis")
# Example usage and testing
if __name__ == "__main__":
    # Load robot poses from config
    config_path = "/home/weini/code/robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
    robot_poses = get_robot_poses(config_path)

    # Create tracker
    tracker = BoxTracker(config_path)

    print("Starting continuous grasping analysis...")
    print("Robot poses loaded. Analyzing box poses in real-time...")

    try:
        while True:
            # Get current box data from pose estimation
            boxes = tracker.get_boxes()

            if boxes:
                # Use first detected box
                box = boxes[0]

                print(f"\n=== Box {box.id} Analysis ===")
                print(
                    f"Position: [{box.position[0]:.3f}, {box.position[1]:.3f}, {box.position[2]:.3f}]"
                )
                print(f"Confidence: {box.confidence:.2f}")

                # Analyze grasping strategy
                best_pair, analysis = find_best_grasping_pair(box, robot_poses)

                # Show all options
                print("All grasping options:")
                for pair_type, data in analysis.items():
                    print(f"  {pair_type}: {data['total_distance']:.3f}m")

                print(f"→ Best: {best_pair.pair_type}")
                best_analysis = analysis[best_pair.pair_type]
                print(f"  Total distance: {best_analysis['total_distance']:.3f}m")
                print(f"  Robot assignment: {best_analysis['robot_assignment']}")
                print(
                    f"  Robot 0 → {best_analysis['distances']['robot0_to_target']:.3f}m"
                )
                print(
                    f"  Robot 1 → {best_analysis['distances']['robot1_to_target']:.3f}m"
                )

            else:
                print(".", end="", flush=True)

            time.sleep(0.5)  # Update every 500ms

    except KeyboardInterrupt:
        print("\nStopping analysis...")
    finally:
        tracker.close()
