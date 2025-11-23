from motion_planner.motion_planner import MotionPlanner
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    planner = MotionPlanner()
    # Base Poses
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

    # Helper function needed to define new position poses
    def make_pose_from_p_R(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Helper to create a 4x4 homogenous pose from position and 3x3 rotation matrix."""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    # Constants
    hz = 100
    dt = 1 / hz
    MOVE_DIST = 0.15 # 15 cm
    MAX_VEL = 0.25
    MAX_ACC = 0.5
    HOLD_DUR = 0.5

    # Define base position for movements (P_base = [0, 0, 0.2])
    P_base = pose_lifted_base[:3, 3].copy()
    R_base = pose_lifted_base[:3, :3].copy() # Identity

    # Define the 4 corner-like poses relative to the lifted base position
    pose_y_plus = make_pose_from_p_R(P_base + [0, MOVE_DIST, 0], R_base)
    pose_x_plus = make_pose_from_p_R(P_base + [MOVE_DIST, 0, 0], R_base)
    
    # Initialize segments list and current pose
    trajectory_segments = []
    current_pose = pose_start.copy()


    # --- 1. Lift Up (Z-axis movement) ---
    seg_lift = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_lift)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))


    # --- 2. Y-Axis Back and Forth (15 cm) ---
    
    # Move +Y
    seg_y_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_y_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_y_plus)
    current_pose = pose_y_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # Move -Y (back to center)
    seg_y_minus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_y_minus)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))


    # --- 3. X-Axis Back and Forth (15 cm) ---
    
    # Move +X
    seg_x_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_x_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_x_plus)
    current_pose = pose_x_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # Move -X (back to center)
    seg_x_minus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_x_minus)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))


    # --- 4. Y-Axis Back and Forth (15 cm) - Second Time ---
    
    # Move +Y
    seg_y_plus_2 = planner.linear(
        start_pose=current_pose,
        end_pose=pose_y_plus,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_y_plus_2)
    current_pose = pose_y_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # Move -Y (back to center)
    seg_y_minus_2 = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
        
    )
    trajectory_segments.append(seg_y_minus_2)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))


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
    linear_motion_test_trajectory = planner.concatenate_trajectories(trajectory_segments)

    # --- Save and Plot ---
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = linear_motion_test_trajectory.plot_3d(show_frames=False)
    fig3d.savefig("plots/trajectory_linear_test_3d.png", dpi=150, bbox_inches="tight")
    

    fig_profiles = linear_motion_test_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_linear_test_profiles.png", dpi=150, bbox_inches="tight")
    

    linear_motion_test_trajectory.save_trajectory("motion_planner/trajectories/linear.npz")