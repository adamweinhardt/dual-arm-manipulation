import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp
from motion_planner.motion_planner import MotionPlanner, Trajectory



class MotionPlanner(MotionPlanner):
    def __init__(self):
        pass

    def figure8_complex(
            self,
            start_pose: np.ndarray,
            dt: float,
            width: float = 0.4,
            height: float = 0.2,
            z_amp: float = 0.05,
            duration: float = 6.0,
            label: str = "figure8_complex"
        ) -> Trajectory:
            p0 = start_pose[:3, 3]
            R0 = start_pose[:3, :3]
            
            traj = Trajectory(label, {"width": width, "duration": duration})
            num_steps = int(duration / dt)
            
            w_pos = 2 * np.pi / duration
            w_roll, w_pitch, w_yaw = w_pos * 2.0, w_pos * 1.5, w_pos * 1.0
            
            Ax, Ay, Az = width/2.0, height/2.0, z_amp
            A_r, A_p, A_y = np.deg2rad(20), np.deg2rad(15), np.deg2rad(30)

            def get_sin(t, A, w, phi=0.0):
                return (A*np.sin(w*t+phi), A*w*np.cos(w*t+phi), -A*w**2*np.sin(w*t+phi))

            for i in range(num_steps + 1):
                t = i * dt
                # Position: Lemniscate + Z bob
                x, vx, ax = get_sin(t, Ax, w_pos)
                y, vy, ay = get_sin(t, Ay, 2*w_pos) 
                z, vz, az = get_sin(t, Az, 2*w_pos, np.pi/2)
                
                pos = p0 + np.array([x, y, z])
                lin_vel, lin_acc = np.array([vx, vy, vz]), np.array([ax, ay, az])
                
                # Rotation: Coupled wobbles
                r, dr, ddr = get_sin(t, A_r, w_roll)
                p, dp, ddp = get_sin(t, A_p, w_pitch)
                y_ang, dy, ddy = get_sin(t, A_y, w_yaw)
                
                R_local = Rotation.from_euler("xyz", [r, p, y_ang])
                rot = R0 @ R_local.as_matrix()
                
                # Angular Velocity (Body -> World approximation)
                sr, cr, sp, cp = np.sin(r), np.cos(r), np.sin(p), np.cos(p)
                wx = dr + dy*sp
                wy = dp*cr - dy*cp*sr
                wz = dp*sr + dy*cp*cr
                
                ang_vel = R0 @ np.array([wx, wy, wz])
                
                # Approx Accel
                alph_x = ddr + ddy*sp
                alph_y = ddp*cr - ddy*cp*sr
                alph_z = ddp*sr + ddy*cp*cr
                ang_acc = R0 @ np.array([alph_x, alph_y, alph_z])

                traj.add_waypoint(pos, rot, lin_vel, ang_vel, lin_acc, ang_acc, i, t)
            return traj

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    planner = MotionPlanner()
    dt = 1/150

    # --- 1. Define Poses ---
    pose_start = np.eye(4)
    # Let's say we start at [0, 0, 0]
    pose_start[:3, 3] = [0.0, 0.0, 0.0]

    # Lifted pose is 0.2m higher
    pose_lifted = pose_start.copy()
    pose_lifted[2, 3] += 0.2

    print("Generating Sequence 1: Lift -> Oscillating Circle -> Return")

    # --- 2. Generate Segments ---

    # Segment A: Lift Up
    traj_up = planner.linear(
        start_pose=pose_start, 
        end_pose=pose_lifted, 
        dt=dt, 
    )

    # Segment B: The Task (Figure 8 Stress Test)
    # Starts at pose_lifted, moves +/- 0.2m in X/Y, +/- 0.05m in Z relative to lift
    traj_figure8 = planner.figure8_complex(
        start_pose=pose_lifted,
        dt=dt,
        width=0.6,       # X range
        height=0.2,      # Y range
        z_amp=0.05,      # Z wobble
        duration=10.0     # Time to complete
    )

    # Segment C: Return
    # The figure 8 logic ensures it returns to (0,0,0) relative to start, 
    # so we are essentially back at pose_lifted.
    traj_down = planner.linear(
        start_pose=pose_lifted,
        end_pose=pose_start,
        dt=dt,
    )

    # --- 2. Concatenate and Save ---
    full_f8_sequence = planner.concatenate_trajectories([traj_up, traj_figure8, traj_down])
    
    full_f8_sequence.save_trajectory("motion_planner/trajectories/figure8_complex.npz")
    
    # Plotting
    fig, _ = full_f8_sequence.plot_3d(show_frames=True)
    fig.savefig("plots/seq_figure8_complex_3d.png")
    plt.close(fig)

    fig = full_f8_sequence.plot_profiles()
    fig.savefig("plots/seq_figure8_complex_profiles.png")
    plt.close(fig)
    print("Done Sequence 2.")