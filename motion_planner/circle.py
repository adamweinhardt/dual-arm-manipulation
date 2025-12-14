import numpy as np
import matplotlib

matplotlib.use("Agg")
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from motion_planner.motion_planner import MotionPlanner, Trajectory


class MotionPlanner(MotionPlanner):
    def __init__(self):
        pass

    def circular_oscillating(
        self,
        start_pose: np.ndarray,
        dt: float,
        radius: float = 0.2,
        max_velocity: float = 0.5,
        max_acceleration: float = 0.25,
        rot_amplitude_deg: float = 45.0,
        rot_cycles: float = 1.0,
        z_amplitude: float = 0.1,
        z_cycles: float = 2.0,
        ccw: bool = True,
        label: str = "circular_rollercoaster",
    ) -> Trajectory:
        p0 = start_pose[:3, 3]
        R0 = start_pose[:3, :3]

        center = p0.copy()
        center[0] -= radius

        s_total = 2 * np.pi * radius

        t_acc = max_velocity / max_acceleration
        s_acc = 0.5 * max_acceleration * t_acc**2

        if s_total < 2 * s_acc:
            t_acc = np.sqrt(s_total / max_acceleration)
            T = 2 * t_acc
            v_peak = max_acceleration * t_acc
        else:
            s_const = s_total - 2 * s_acc
            t_const = s_const / max_velocity
            T = 2 * t_acc + t_const
            v_peak = max_velocity

        traj = Trajectory(
            label,
            {"radius": radius, "z_amp": z_amplitude, "rot_deg": rot_amplitude_deg},
        )

        num_steps = int(np.ceil(T / dt))
        dir_sign = 1.0 if ccw else -1.0

        amp_rot_rad = np.deg2rad(rot_amplitude_deg)

        for i in range(num_steps + 1):
            t = min(i * dt, T)

            if t < t_acc:
                s = 0.5 * max_acceleration * t**2
                s_dot = max_acceleration * t
                s_ddot = max_acceleration
            elif t < T - t_acc:
                s = s_acc + v_peak * (t - t_acc)
                s_dot = v_peak
                s_ddot = 0.0
            else:
                td = T - t
                s = s_total - 0.5 * max_acceleration * td**2
                s_dot = max_acceleration * td
                s_ddot = -max_acceleration

            phi = s / radius
            phi_dot = s_dot / radius
            phi_ddot = s_ddot / radius

            theta = dir_sign * phi

            c, si = np.cos(theta), np.sin(theta)
            x = center[0] + radius * c
            y = center[1] + radius * si

            vx = -radius * si * (dir_sign * phi_dot)
            vy = radius * c * (dir_sign * phi_dot)
            ax = -radius * c * (phi_dot**2) - radius * si * (dir_sign * phi_ddot)
            ay = -radius * si * (phi_dot**2) + radius * c * (dir_sign * phi_ddot)

            z_phase = z_cycles * phi

            z = p0[2] + z_amplitude * np.sin(z_phase)

            z_dot = z_amplitude * z_cycles * np.cos(z_phase) * phi_dot

            term1 = -z_amplitude * (z_cycles**2) * np.sin(z_phase) * (phi_dot**2)
            term2 = z_amplitude * z_cycles * np.cos(z_phase) * phi_ddot
            z_ddot = term1 + term2

            position = np.array([x, y, z])
            lin_vel = np.array([vx, vy, z_dot])
            lin_acc = np.array([ax, ay, z_ddot])

            rot_phase = rot_cycles * phi

            yaw_val = amp_rot_rad * np.sin(rot_phase)

            yaw_dot = amp_rot_rad * rot_cycles * np.cos(rot_phase) * phi_dot

            y_term1 = -amp_rot_rad * (rot_cycles**2) * np.sin(rot_phase) * (phi_dot**2)
            y_term2 = amp_rot_rad * rot_cycles * np.cos(rot_phase) * phi_ddot
            yaw_ddot = y_term1 + y_term2

            R_osc = Rotation.from_euler("z", yaw_val).as_matrix()
            rot_matrix = R0 @ R_osc

            ang_vel = R0 @ np.array([0, 0, yaw_dot])
            ang_acc = R0 @ np.array([0, 0, yaw_ddot])

            traj.add_waypoint(
                position, rot_matrix, lin_vel, ang_vel, lin_acc, ang_acc, i, t
            )

        return traj


if __name__ == "__main__":
    import os

    planner = MotionPlanner()
    hz = 100
    dt = 1 / hz

    pose_start = np.eye(4)
    pose_start[:3, 3] = [0.0, 0.0, 0.0]

    pose_lifted = pose_start.copy()
    pose_lifted[2, 3] = 0.2

    traj_up = planner.linear(
        start_pose=pose_start,
        end_pose=pose_lifted,
        dt=dt,
    )

    traj_circle = planner.circular_oscillating(
        start_pose=pose_lifted,
        dt=dt,
        radius=0.17,
        max_velocity=0.3,
        max_acceleration=0.15,
        rot_amplitude_deg=20.0,
        rot_cycles=1.0,
        z_amplitude=0.05,
        z_cycles=0,
    )

    traj_down = planner.linear(
        start_pose=pose_lifted,
        end_pose=pose_start,
        dt=dt,
    )

    full_sequence = planner.concatenate_trajectories([traj_up, traj_circle, traj_down])

    full_sequence.save_trajectory(
        "motion_planner/trajectories/rollercoaster_circle.npz"
    )

    fig, axes = full_sequence.plot_3d(show_frames=True)
    fig.suptitle(f"Rollercoaster Circle (Z-Amp=0.1m, Rot-Cycles=1)")
    fig.savefig("plots/rollercoaster_3d.png")

    full_sequence.plot_profiles().savefig("plots/rollercoaster_profiles.png")
    print("Done generating rollercoaster trajectory.")
