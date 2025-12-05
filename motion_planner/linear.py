from motion_planner.motion_planner import MotionPlanner
import numpy as np

if __name__ == "__main__":
    planner = MotionPlanner()

    # --- Base Poses ---
    pose_start = np.array(  # Identity, z = 0
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    pose_lifted_base = np.array(  # Identity, z = 0.2
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    # Helper: build pose from position + rotation
    def make_pose_from_p_R(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    # --- Constants ---
    hz = 100
    dt = 1.0 / hz
    MOVE_DIST = 0.20   # 20 cm
    HOLD_DUR = 0.2     # seconds

    # Base position/orientation for motions
    P_base = pose_lifted_base[:3, 3].copy()   # [0, 0, 0.2]
    R_base = pose_lifted_base[:3, :3].copy()  # Identity

    # --- Target poses for motions from lifted base ---
    pose_y_plus = make_pose_from_p_R(P_base + np.array([0.0,  MOVE_DIST, 0.0]), R_base)
    pose_z_plus = make_pose_from_p_R(P_base + np.array([0.0,  0.0,        MOVE_DIST]), R_base)
    pose_x_plus = make_pose_from_p_R(P_base + np.array([MOVE_DIST, 0.0,   0.0]),      R_base)

    # --- Build trajectory segments in order ---
    trajectory_segments = []
    current_pose = pose_start.copy()

    # 1) Lift up (start -> lifted base)
    seg_lift = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt
    )
    trajectory_segments.append(seg_lift)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # 2) Y+ then back
    seg_y_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_y_plus,
        dt=dt
    )
    trajectory_segments.append(seg_y_plus)
    current_pose = pose_y_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    seg_y_back = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt
    )
    trajectory_segments.append(seg_y_back)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # 3) Z+ then back
    seg_z_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_z_plus,
        dt=dt
    )
    trajectory_segments.append(seg_z_plus)
    current_pose = pose_z_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    seg_z_back = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt
    )
    trajectory_segments.append(seg_z_back)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # 4) X+ then back
    seg_x_plus = planner.linear(
        start_pose=current_pose,
        end_pose=pose_x_plus,
        dt=dt
    )
    trajectory_segments.append(seg_x_plus)
    current_pose = pose_x_plus.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    seg_x_back = planner.linear(
        start_pose=current_pose,
        end_pose=pose_lifted_base,
        dt=dt
    )
    trajectory_segments.append(seg_x_back)
    current_pose = pose_lifted_base.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # 5) Put down (lifted base -> start)
    seg_down = planner.linear(
        start_pose=current_pose,
        end_pose=pose_start,
        dt=dt
    )
    trajectory_segments.append(seg_down)
    current_pose = pose_start.copy()
    trajectory_segments.append(planner.hold(current_pose, duration=HOLD_DUR, dt=dt))

    # --- Concatenate full trajectory ---
    linear_motion_test_trajectory = planner.concatenate_trajectories(trajectory_segments)

    # --- Save and plot ---
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = linear_motion_test_trajectory.plot_3d(show_frames=False)
    fig3d.savefig("plots/trajectory_linear_test_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = linear_motion_test_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_linear_test_profiles.png", dpi=150, bbox_inches="tight")

    linear_motion_test_trajectory.save_trajectory("motion_planner/trajectories/linear.npz")
