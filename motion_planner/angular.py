from motion_planner.motion_planner import MotionPlanner
import numpy as np
import os

if __name__ == "__main__":
    planner = MotionPlanner()

    pose_start = np.array(  # Identity, z = 0
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    pose_lifted = np.array(  # Identity, z = 0.2
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    def make_pose_from_p_R(p: np.ndarray, R: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    hz = 100
    dt = 1.0 / hz
    HOLD = 0.2
    angle_deg = 30.0
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    p_lift = pose_lifted[:3, 3].copy()  # [0, 0, 0.2]
    R_identity = np.eye(3)

    Rz_plus = np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )
    pose_z_plus = make_pose_from_p_R(p_lift, Rz_plus)
    pose_z_base = make_pose_from_p_R(p_lift, R_identity)

    Ry_plus = np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )
    pose_y_plus = make_pose_from_p_R(p_lift, Ry_plus)
    pose_y_base = make_pose_from_p_R(p_lift, R_identity)

    Rx_plus = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )
    pose_x_plus = make_pose_from_p_R(p_lift, Rx_plus)
    pose_x_base = make_pose_from_p_R(p_lift, R_identity)

    segments = []

    seg_lift = planner.linear(
        start_pose=pose_start,
        end_pose=pose_lifted,
        dt=dt,
    )
    segments.append(seg_lift)
    segments.append(planner.hold(pose_lifted, duration=HOLD, dt=dt))

    seg_z_plus = planner.linear(
        start_pose=pose_lifted,
        end_pose=pose_z_plus,
        dt=dt,
    )
    segments.append(seg_z_plus)
    segments.append(planner.hold(pose_z_plus, duration=HOLD, dt=dt))

    seg_z_back = planner.linear(
        start_pose=pose_z_plus,
        end_pose=pose_z_base,
        dt=dt,
    )
    segments.append(seg_z_back)
    segments.append(planner.hold(pose_z_base, duration=HOLD, dt=dt))

    seg_y_plus = planner.linear(
        start_pose=pose_y_base,
        end_pose=pose_y_plus,
        dt=dt,
    )
    segments.append(seg_y_plus)
    segments.append(planner.hold(pose_y_plus, duration=HOLD, dt=dt))

    seg_y_back = planner.linear(
        start_pose=pose_y_plus,
        end_pose=pose_y_base,
        dt=dt,
    )
    segments.append(seg_y_back)
    segments.append(planner.hold(pose_y_base, duration=HOLD, dt=dt))

    seg_x_plus = planner.linear(
        start_pose=pose_x_base,
        end_pose=pose_x_plus,
        dt=dt,
    )
    segments.append(seg_x_plus)
    segments.append(planner.hold(pose_x_plus, duration=HOLD, dt=dt))

    seg_x_back = planner.linear(
        start_pose=pose_x_plus,
        end_pose=pose_x_base,
        dt=dt,
    )
    segments.append(seg_x_back)
    segments.append(planner.hold(pose_x_base, duration=HOLD, dt=dt))

    seg_down = planner.linear(
        start_pose=pose_x_base,
        end_pose=pose_start,
        dt=dt,
    )
    segments.append(seg_down)
    segments.append(planner.hold(pose_start, duration=HOLD, dt=dt))

    rotation_test_trajectory = planner.concatenate_trajectories(segments)

    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = rotation_test_trajectory.plot_3d(show_frames=True)
    fig3d.savefig("plots/trajectory_rotation_test_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = rotation_test_trajectory.plot_profiles()
    fig_profiles.savefig(
        "plots/trajectory_rotation_test_profiles.png", dpi=150, bbox_inches="tight"
    )

    rotation_test_trajectory.save_trajectory("motion_planner/trajectories/angular.npz")
