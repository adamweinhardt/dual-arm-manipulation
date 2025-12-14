from motion_planner.motion_planner import MotionPlanner
import numpy as np

if __name__ == "__main__":
    planner = MotionPlanner()
import os

if __name__ == "__main__":
    planner = MotionPlanner()

    pose1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    pose2 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],  # lift 0.2 m
            [0, 0, 0, 1],
        ]
    )

    theta = np.deg2rad(0)
    c, s = np.cos(theta), np.sin(theta)
    pose3 = np.array(
        [
            [c, -s, 0, -1],
            [s, c, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    pose4 = np.array(
        [
            [c, -s, 0, -1],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    hz = 100
    dt = 1 / hz

    factor = 0.8

    max_lin_vel = 0.5 * factor
    max_lin_acc = 0.25 * factor
    max_ang_vel = 0.5 * factor
    max_ang_acc = 0.25 * factor

    up = planner.linear(
        pose1,
        pose2,
        dt,
        max_lin_vel=max_lin_vel,
        max_lin_acc=max_lin_acc,
        max_ang_vel=max_ang_vel,
        max_ang_acc=max_ang_acc,
    )
    transport_with_yaw = planner.linear(
        pose2,
        pose3,
        dt,
        max_lin_vel=max_lin_vel,
        max_lin_acc=max_lin_acc,
        max_ang_vel=max_ang_vel,
        max_ang_acc=max_ang_acc,
    )
    place_down_with_yaw = planner.linear(
        pose3,
        pose4,
        dt,
        max_lin_vel=max_lin_vel,
        max_lin_acc=max_lin_acc,
        max_ang_vel=max_ang_vel,
        max_ang_acc=max_ang_acc,
    )

    full_trajectory = planner.concatenate_trajectories(
        [up, transport_with_yaw, place_down_with_yaw]
    )

    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    fig3d, _ = full_trajectory.plot_3d()
    fig3d.savefig("plots/trajectory_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = full_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_profiles.png", dpi=150, bbox_inches="tight")

    full_trajectory.save_trajectory(
        f"motion_planner/trajectories/pick_and_place_{max_lin_vel}v_{max_lin_acc}a_{max_ang_vel}w_{max_ang_acc}B_{hz}Hz_weight.npz"
    )
