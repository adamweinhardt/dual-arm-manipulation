from motion_planner.motion_planner import MotionPlanner
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    planner = MotionPlanner()
    # Base Poses from your script:
    pose_start = np.array(  # Identity, z=0
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    pose_lifted_base = np.array(  # Identity, z=0.2
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    # Helper function needed for the rotation logic
    def make_pose_from_p_R(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Helper to create a 4x4 homogenous pose from position and 3x3 rotation matrix."""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    # Constants
    hz = 100
    dt = 1 / hz
    rot_angle_deg = 30.0
    rot_angle_rad = np.deg2rad(rot_angle_deg)
    MAX_VEL = 0.25 # m/s or rad/s equivalent
    MAX_ACC = 0.5 # m/s^2 or rad/s^2 equivalent

    planner = MotionPlanner()
    trajectory_segments = []

    # --- 1. Lift Up (Z-axis movement) ---
    seg_lift = planner.linear(
        start_pose=pose_start,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_lift)

    # Hold at lifted position
    current_pose = pose_lifted_base.copy()
    hold_seg = planner.hold(current_pose, duration=0.5, dt=dt)
    trajectory_segments.append(hold_seg)

    # --- 2. Z-Axis Rotation (+30 deg, then -30 deg) ---
    R_base = current_pose[:3, :3]
    P_base = current_pose[:3, 3]

    # Target 1: +30 deg Yaw (Z)
    R_Z_plus = Rotation.from_euler("z", rot_angle_rad).as_matrix()
    R_target_Z_plus = R_Z_plus @ R_base
    pose_Z_plus = make_pose_from_p_R(P_base, R_target_Z_plus)

    seg_Z_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_Z_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_Z_plus)
    current_pose = pose_Z_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))

    # Target 2: Back to base orientation
    seg_Z_minus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,  # Go back to the original orientation
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_Z_minus)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))


    # --- 3. Y-Axis Rotation (+30 deg, then -30 deg) ---
    R_base = current_pose[:3, :3] # Should be Identity again
    P_base = current_pose[:3, 3]

    # Target 3: +30 deg Pitch (Y)
    R_Y_plus = Rotation.from_euler("y", rot_angle_rad).as_matrix()
    R_target_Y_plus = R_Y_plus @ R_base
    pose_Y_plus = make_pose_from_p_R(P_base, R_target_Y_plus)

    seg_Y_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_Y_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_Y_plus)
    current_pose = pose_Y_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))

    # Target 4: Back to base orientation
    seg_Y_minus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_Y_minus)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))


    # --- 4. X-Axis Rotation (+30 deg, then -30 deg) ---
    R_base = current_pose[:3, :3] # Should be Identity again
    P_base = current_pose[:3, 3]

    # Target 5: +30 deg Roll (X)
    R_X_plus = Rotation.from_euler("x", rot_angle_rad).as_matrix()
    R_target_X_plus = R_X_plus @ R_base
    pose_X_plus = make_pose_from_p_R(P_base, R_target_X_plus)

    seg_X_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_X_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_X_plus)
    current_pose = pose_X_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))

    # Target 6: Back to base orientation
    seg_X_minus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_X_minus)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=0.5, dt=dt))


    # --- 5. Put Down (Z-axis movement) ---
    seg_down = planner.linear(
        start_pose=current_pose,
        end_pose=pose_start,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC,
    )
    trajectory_segments.append(seg_down)

    # --- Concatenate Trajectories ---
    rotation_test_trajectory = planner.concatenate_trajectories(trajectory_segments)

    # --- Save and Plot ---
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = rotation_test_trajectory.plot_3d(show_frames=True)
    fig3d.savefig("plots/trajectory_rotation_test_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = rotation_test_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_rotation_test_profiles.png", dpi=150, bbox_inches="tight")

    rotation_test_trajectory.save_trajectory("motion_planner/trajectories/twist.npz")