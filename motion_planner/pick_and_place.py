from motion_planner.motion_planner import MotionPlanner
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    planner = MotionPlanner()

    # --- Configuration ---
    LIFT_HEIGHT = 0.20  # 20 cm Z-lift
    PLACE_Y_OFFSET = -0.50 # 50 cm move in -Y direction
    MAX_VEL = 0.25 # m/s
    MAX_ACC = 0.5  # m/s^2
    HOLD_DUR = 0.5 # seconds to pause
    hz = 100
    dt = 1 / hz

    # --- Helper function for Pose Creation ---
    def make_pose_from_p_R(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Helper to create a 4x4 homogenous pose from position and 3x3 rotation matrix."""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    # --- Define Key Poses (All with Identity Rotation) ---

    # 1. Start/Pick Pose (0, 0, 0)
    pose_pick_start = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    R_base = pose_pick_start[:3, :3]

    # 2. Lifted Pose (0, 0, 0.2)
    pose_lifted_pick = make_pose_from_p_R(
        np.array([0.0, 0.0, LIFT_HEIGHT]), R_base
    )

    # 3. Lifted Place Pose (0, -0.5, 0.2)
    pose_lifted_place = make_pose_from_p_R(
        np.array([0.0, PLACE_Y_OFFSET, LIFT_HEIGHT]), R_base
    )
    
    # 4. Final Place Pose (0, -0.5, 0.0)
    pose_place_end = make_pose_from_p_R(
        np.array([0.0, PLACE_Y_OFFSET, 0.0]), R_base
    )

    # --- Sequence Generation ---
    trajectory_segments = []
    current_pose = pose_pick_start.copy()

    # 1. Lift Up (Pick Approach)
    seg_lift_up = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_pick,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_lift_up)
    current_pose = pose_lifted_pick.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt)) # Pause to simulate grasping


    # 2. Move to Place Location (-Y 50 cm)
    seg_move_place = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_place,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_move_place)
    current_pose = pose_lifted_place.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt)) # Pause to simulate releasing


    # 3. Put Down (Place Descent)
    seg_put_down = planner.linear(
        start_pose=current_pose,
        end_pose=pose_place_end,
        dt=dt,
        max_velocity=MAX_VEL,
        max_acceleration=MAX_ACC
    )
    trajectory_segments.append(seg_put_down)
    current_pose = pose_place_end.copy()


    # --- Concatenate Trajectories ---
    pick_and_place_trajectory = planner.concatenate_trajectories(trajectory_segments)

    # --- Save and Plot ---
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = pick_and_place_trajectory.plot_3d(show_frames=False)
    fig3d.savefig("plots/trajectory_pick_and_place_3d.png", dpi=150, bbox_inches="tight")
    

    fig_profiles = pick_and_place_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_pick_and_place_profiles.png", dpi=150, bbox_inches="tight")
    
    pick_and_place_trajectory.save_trajectory("motion_planner/trajectories/pick_and_place.npz")