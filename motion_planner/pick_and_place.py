from motion_planner.motion_planner import MotionPlanner
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    planner = MotionPlanner()

import numpy as np
import os

if __name__ == "__main__":
    planner = MotionPlanner()

    # Base → lift-up (same as your original)
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
            [0, 0, 1, 0.2],   # lift 0.2 m
            [0, 0, 0, 1],
        ]
    )

    # --- Add 30° rotation about z for the transport & place ---
    theta = np.deg2rad(45.0)
    c, s = np.cos(theta), np.sin(theta)
    pose3 = np.array(
        [
            [ c, -s, 0, -1],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0.2],
            [ 0,  0, 0, 1],
        ]
    )

    # Transport target (x = -1 m, z = 0.2 m) with final 30° yaw
    pose4= np.array(
        [
            [ c, -s, 0, -1],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1],
        ]
    )

    # ---- Timing & segments (tuned like your original) ----
    hz = 100
    dt = 1 / hz

    max_lin_vel = 0.5  # m/s
    max_lin_acc = 0.25  # m/s²
    max_ang_vel = 0.5 #rad/s
    max_ang_acc = 0.25 #rad/s²


    up = planner.linear(pose1, pose2, dt, max_lin_vel = max_lin_vel, max_lin_acc = max_lin_acc, max_ang_vel = max_ang_vel,max_ang_acc = max_ang_acc)
    transport_with_yaw = planner.linear(
        pose2, pose3, dt, max_lin_vel = max_lin_vel, max_lin_acc = max_lin_acc, max_ang_vel = max_ang_vel,max_ang_acc = max_ang_acc
    )
    place_down_with_yaw = planner.linear(
        pose3, pose4, dt, max_lin_vel = max_lin_vel, max_lin_acc = max_lin_acc, max_ang_vel = max_ang_vel,max_ang_acc = max_ang_acc
    )

    full_trajectory = planner.concatenate_trajectories(
        [up, transport_with_yaw, place_down_with_yaw]
    )

    # Ensure directory exists BEFORE saving figures/files
    os.makedirs("plots", exist_ok=True)
    os.makedirs("motion_planner/trajectories", exist_ok=True)

    # Plots
    fig3d, _ = full_trajectory.plot_3d()
    fig3d.savefig("plots/trajectory_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = full_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_profiles.png", dpi=150, bbox_inches="tight")

    # Save trajectory bundle
    full_trajectory.save_trajectory(
        f"motion_planner/trajectories/pick_and_place_{max_lin_vel}v_{max_lin_acc}a_{max_ang_vel}w_{max_ang_acc}B.npz"
    )
