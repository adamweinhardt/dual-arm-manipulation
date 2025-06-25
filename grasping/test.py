import numpy as np
import sys
import os
import json
import time
from scipy.spatial.transform import Rotation as R

# Import the BoardPoseEstimator
sys.path.append("robot_ipc_control")
from robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator


def debug_coordinate_system():
    """Debug the entire coordinate system pipeline"""

    config_path = (
        "robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    port = config.get("port", 5557)

    print("=== COMPREHENSIVE COORDINATE DEBUG ===")
    print(f"Using port: {port}")

    # Initialize pose estimator
    pose_estimator = BoardPoseEstimator(f"tcp://localhost:{port}")
    pose_estimator.start()

    time.sleep(2)  # Let it collect data

    tracked_ids = pose_estimator.get_tracked_board_ids()
    print(f"Tracked IDs: {tracked_ids}")

    if not tracked_ids:
        print("No tracked boxes found!")
        pose_estimator.stop()
        return

    board_id = tracked_ids[0]
    print(f"\nAnalyzing board {board_id}")

    # Get pose data
    pose = pose_estimator.get_pose(board_id)
    confidence = pose_estimator.get_confidence(board_id)
    is_stable = pose_estimator.is_stable(board_id)

    print(f"Pose: {pose}")
    print(f"Confidence: {confidence}")
    print(f"Stable: {is_stable}")

    if pose is None:
        print("No pose data available!")
        pose_estimator.stop()
        return

    # Extract position and rotation
    position = pose[:3]
    quaternion = pose[3:7]
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    print(f"\nPosition: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
    print(
        f"Quaternion: [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]"
    )
    print(f"Rotation matrix:")
    for i in range(3):
        print(
            f"  [{rotation_matrix[i, 0]:+.6f}, {rotation_matrix[i, 1]:+.6f}, {rotation_matrix[i, 2]:+.6f}]"
        )

    # Load box dimensions
    config_dir = os.path.dirname(os.path.dirname(config_path))
    board_config = config["boards"][0]
    if not os.path.isabs(board_config):
        board_config = os.path.join(config_dir, board_config.lstrip("./"))

    with open(board_config, "r") as f:
        board_data = json.load(f)

    # Calculate dimensions
    min_pt = np.array([float("inf")] * 3)
    max_pt = np.array([float("-inf")] * 3)

    for marker in board_data["markers"]:
        for corner in marker["corners"]:
            for k in range(3):
                min_pt[k] = min(min_pt[k], corner[k])
                max_pt[k] = max(max_pt[k], corner[k])

    dims = max_pt - min_pt
    w, h, d = dims[0], dims[1], dims[2]

    print(f"\nBox dimensions: W={w:.6f}, H={h:.6f}, D={d:.6f}")
    print(
        f"Box center (from pose): [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]"
    )

    # Test different face center calculations
    print(f"\n=== FACE CENTER TESTS ===")

    # Version 1: Current implementation
    faces_v1 = {
        "top": np.array([0, 0, +d / 2]),
        "bottom": np.array([0, 0, -d / 2]),
        "front": np.array([+w / 2, 0, 0]),
        "back": np.array([-w / 2, 0, 0]),
        "right": np.array([0, -h / 2, 0]),
        "left": np.array([0, +h / 2, 0]),
    }

    # Version 2: Flipped Z
    faces_v2 = {
        "top": np.array([0, 0, -d / 2]),
        "bottom": np.array([0, 0, +d / 2]),
        "front": np.array([+w / 2, 0, 0]),
        "back": np.array([-w / 2, 0, 0]),
        "right": np.array([0, -h / 2, 0]),
        "left": np.array([0, +h / 2, 0]),
    }

    # Version 3: Different Y assignment
    faces_v3 = {
        "top": np.array([0, 0, +d / 2]),
        "bottom": np.array([0, 0, -d / 2]),
        "front": np.array([+w / 2, 0, 0]),
        "back": np.array([-w / 2, 0, 0]),
        "right": np.array([0, +h / 2, 0]),  # Flipped
        "left": np.array([0, -h / 2, 0]),  # Flipped
    }

    # Version 4: Different axis assignment (Y as up)
    faces_v4 = {
        "top": np.array([0, +d / 2, 0]),  # Y as up
        "bottom": np.array([0, -d / 2, 0]),
        "front": np.array([+w / 2, 0, 0]),
        "back": np.array([-w / 2, 0, 0]),
        "right": np.array([0, 0, +h / 2]),
        "left": np.array([0, 0, -h / 2]),
    }

    versions = [
        ("V1 (Current)", faces_v1),
        ("V2 (Flipped Z)", faces_v2),
        ("V3 (Flipped Y)", faces_v3),
        ("V4 (Y as Up)", faces_v4),
    ]

    for version_name, faces_local in versions:
        print(f"\n{version_name}:")
        for name, local_pos in faces_local.items():
            world_pos = rotation_matrix @ local_pos + position
            print(
                f"  {name:6}: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]"
            )

        # Check which version gives reasonable Z values
        top_z = (rotation_matrix @ faces_local["top"] + position)[2]
        bottom_z = (rotation_matrix @ faces_local["bottom"] + position)[2]
        print(f"  → Top Z: {top_z:.3f}, Bottom Z: {bottom_z:.3f}")

        if top_z > bottom_z and bottom_z > 0:
            print(f"  ✓ This version looks reasonable!")
        elif top_z > bottom_z:
            print(f"  ~ Top > Bottom (good) but bottom below table")
        elif bottom_z > 0:
            print(f"  ~ Both above table but top < bottom (bad)")
        else:
            print(f"  ✗ This version looks wrong")

    print(f"\n=== BOARD CONFIG ANALYSIS ===")
    print(f"Board markers min point: {min_pt}")
    print(f"Board markers max point: {max_pt}")
    print(f"Board coordinate span: {dims}")

    # Check if board has Z variation
    z_coords = []
    for marker in board_data["markers"]:
        for corner in marker["corners"]:
            z_coords.append(corner[2])

    z_min, z_max = min(z_coords), max(z_coords)
    print(f"Board Z range: {z_min:.6f} to {z_max:.6f} (span: {z_max - z_min:.6f})")

    if abs(z_max - z_min) < 0.001:
        print("Board is flat (Z variation < 1mm)")
    else:
        print("Board has significant Z variation - might be 3D")

    pose_estimator.stop()


if __name__ == "__main__":
    debug_coordinate_system()
