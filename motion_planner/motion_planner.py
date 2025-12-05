import numpy as np
import matplotlib

matplotlib.use("Agg")  # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Waypoint:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    index: int
    time: float

    def to_dict(self):
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "linear_velocity": self.linear_velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "linear_acceleration": self.linear_acceleration.tolist(),
            "angular_acceleration": self.angular_acceleration.tolist(),
            "index": self.index,
            "time": self.time,
        }


class Trajectory:
    def __init__(self, primitive_type: str, parameters: Dict[str, Any]):
        self.waypoints: List[Waypoint] = []
        self.primitive_type = primitive_type
        self.parameters = parameters

    def add_waypoint(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        angular_acceleration: np.ndarray,
        index: int,
        time: float,
    ):
        """Add a waypoint to the trajectory"""
        self.waypoints.append(
            Waypoint(
                position,
                rotation,
                linear_velocity,
                angular_velocity,
                linear_acceleration,
                angular_acceleration,
                index,
                time,
            )
        )

    def get_positions(self):
        return np.array([wp.position for wp in self.waypoints])

    def get_rotations(self):
        return np.array([wp.rotation for wp in self.waypoints])

    def get_linear_velocities(self):
        return np.array([wp.linear_velocity for wp in self.waypoints])

    def get_angular_velocities(self):
        return np.array([wp.angular_velocity for wp in self.waypoints])

    def get_linear_accelerations(self):
        return np.array([wp.linear_acceleration for wp in self.waypoints])

    def get_angular_accelerations(self):
        return np.array([wp.angular_acceleration for wp in self.waypoints])

    def save_trajectory(self, filename: str):
        if not filename.endswith(".npz"):
            filename += ".npz"

        # Prepare data
        positions = self.get_positions()
        rotations = self.get_rotations()
        velocities = self.get_linear_velocities()
        angular_velocities = self.get_angular_velocities()
        accelerations = self.get_linear_accelerations()
        angular_accelerations = self.get_angular_accelerations()
        times = np.array([wp.time for wp in self.waypoints])

        euler_angles = []
        for rot in rotations:
            r = Rotation.from_matrix(rot)
            euler_angles.append(r.as_euler("xyz"))
        euler_angles = np.array(euler_angles)

        np.savez_compressed(
            filename,
            trajectory_type=self.primitive_type,
            parameters=self.parameters,
            time=times,
            position=positions,
            rotation_matrices=rotations,
            euler_angles=euler_angles,
            linear_velocity=velocities,
            angular_velocity=angular_velocities,
            linear_acceleration=accelerations,
            angular_acceleration=angular_accelerations,
        )

        print(f"Trajectory saved to {filename}")
        print(f"Total waypoints: {len(self.waypoints)}")
        print(f"Total time: {times[-1]:.3f}s" if len(times) > 0 else "No waypoints")

    @classmethod
    def load_trajectory(cls, filename: str):
        """Load trajectory from NPZ file

        Args:
            filename: Input filename
        Returns:
            Trajectory object
        """
        # Add .npz extension if not present
        if not filename.endswith(".npz"):
            filename += ".npz"

        data = np.load(filename, allow_pickle=True)

        trajectory = cls(
            str(data["trajectory_type"]),
            data["parameters"].item() if "parameters" in data else {},
        )

        times = data["time"]
        positions = data["position"]
        rotations = data["rotation_matrices"]
        velocities = data["linear_velocity"]
        angular_velocities = data["angular_velocity"]
        accelerations = data["linear_acceleration"]
        angular_accelerations = data["angular_acceleration"]

        for i in range(len(times)):
            trajectory.add_waypoint(
                positions[i],
                rotations[i],
                velocities[i],
                angular_velocities[i],
                accelerations[i],
                angular_accelerations[i],
                i,
                times[i],
            )

        print(f"Trajectory loaded from {filename}")
        print(f"Loaded {len(trajectory.waypoints)} waypoints")
        return trajectory

    def plot_3d(self, show_frames=False):
        """Plot 3D trajectory with position, velocity, and acceleration heatmaps"""
        frame_scale = 0.02

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(18, 6))

        # Get data
        positions = self.get_positions()
        velocities = self.get_linear_velocities()
        accelerations = self.get_linear_accelerations()

        # Calculate magnitudes for coloring
        vel_magnitudes = np.linalg.norm(velocities, axis=1)  # Regular magnitude
        acc_magnitudes = np.linalg.norm(
            accelerations, axis=1
        )  # Regular magnitude for acceleration too

        # Calculate common axis limits for all subplots
        max_range = (
            np.array(
                [
                    positions[:, 0].max() - positions[:, 0].min(),
                    positions[:, 1].max() - positions[:, 1].min(),
                    positions[:, 2].max() - positions[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        # Subplot 1: Position trajectory
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            "b-",
            linewidth=2,
            label=f"{self.primitive_type} trajectory",
        )

        ax1.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="green",
            s=100,
            label="Start",
        )
        ax1.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="red",
            s=100,
            label="End",
        )

        if show_frames:
            step = max(1, len(self.waypoints) // 10)
            for i in range(0, len(self.waypoints), step):
                wp = self.waypoints[i]
                pos = wp.position
                rot = wp.rotation

                # Draw coordinate frame
                x_axis = rot[:, 0] * frame_scale
                y_axis = rot[:, 1] * frame_scale
                z_axis = rot[:, 2] * frame_scale

                ax1.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    x_axis[0],
                    x_axis[1],
                    x_axis[2],
                    color="red",
                    alpha=0.7,
                )
                ax1.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    y_axis[0],
                    y_axis[1],
                    y_axis[2],
                    color="green",
                    alpha=0.7,
                )
                ax1.quiver(
                    pos[0],
                    pos[1],
                    pos[2],
                    z_axis[0],
                    z_axis[1],
                    z_axis[2],
                    color="blue",
                    alpha=0.7,
                )

        # Add frame points
        step = max(1, len(self.waypoints) // 50)
        frame_positions = []
        for i in range(0, len(self.waypoints), step):
            frame_positions.append(self.waypoints[i].position)

        if frame_positions:
            frame_positions = np.array(frame_positions)
            ax1.scatter(
                frame_positions[:, 0],
                frame_positions[:, 1],
                frame_positions[:, 2],
                c="black",
                s=30,
                alpha=0.8,
                label="Frame points",
            )

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.legend()
        ax1.set_title("Position Trajectory")
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)

        # Subplot 2: Velocity heatmap
        ax2 = fig.add_subplot(132, projection="3d")

        # Create line segments for velocity coloring
        for i in range(len(positions) - 1):
            # Use average velocity magnitude between two waypoints for coloring
            vel_avg = (vel_magnitudes[i] + vel_magnitudes[i + 1]) / 2

            # Normalize color to [0, 1] range
            if vel_magnitudes.max() > 0:
                normalized_color = vel_avg / vel_magnitudes.max()
            else:
                normalized_color = 0

            # Plot line segment with color
            ax2.plot(
                [positions[i, 0], positions[i + 1, 0]],
                [positions[i, 1], positions[i + 1, 1]],
                [positions[i, 2], positions[i + 1, 2]],
                color=plt.cm.jet(normalized_color),
                linewidth=3,
            )

        # Add start/end points
        ax2.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="green",
            s=100,
            label="Start",
        )
        ax2.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="red",
            s=100,
            label="End",
        )

        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title("Velocity Magnitude")
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add colorbar for velocity
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.jet,
            norm=plt.Normalize(
                vmin=0,  # Start from 0 for magnitude
                vmax=vel_magnitudes.max(),
            ),
        )
        sm.set_array([])
        cbar1 = plt.colorbar(sm, ax=ax2, shrink=0.5, aspect=20)
        cbar1.set_label("Velocity Magnitude (m/s)")

        # Subplot 3: Acceleration heatmap
        ax3 = fig.add_subplot(133, projection="3d")

        # Create line segments for acceleration coloring
        for i in range(len(positions) - 1):
            # Use average acceleration magnitude between two waypoints for coloring
            acc_avg = (acc_magnitudes[i] + acc_magnitudes[i + 1]) / 2

            # Normalize color to [0, 1] range
            if acc_magnitudes.max() > 0:
                normalized_color = acc_avg / acc_magnitudes.max()
            else:
                normalized_color = 0

            # Plot line segment with color
            ax3.plot(
                [positions[i, 0], positions[i + 1, 0]],
                [positions[i, 1], positions[i + 1, 1]],
                [positions[i, 2], positions[i + 1, 2]],
                color=plt.cm.jet(normalized_color),
                linewidth=3,
            )

        # Add start/end points
        ax3.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="green",
            s=100,
            label="Start",
        )
        ax3.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="red",
            s=100,
            label="End",
        )

        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Z (m)")
        ax3.set_title("Acceleration Magnitude")
        ax3.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add colorbar for acceleration
        sm2 = plt.cm.ScalarMappable(
            cmap=plt.cm.jet,
            norm=plt.Normalize(
                vmin=0,  # Start from 0 for magnitude
                vmax=acc_magnitudes.max(),
            ),
        )
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax3, shrink=0.5, aspect=20)
        cbar2.set_label("Acceleration Magnitude (m/s²)")

        plt.tight_layout()
        return fig, (ax1, ax2, ax3)

    def plot_profiles(self):
        """Plot position, rotation, velocity, acceleration profiles"""
        positions = self.get_positions()
        rotations = self.get_rotations()
        velocities = self.get_linear_velocities()
        angular_velocities = self.get_angular_velocities()
        accelerations = self.get_linear_accelerations()
        angular_accelerations = self.get_angular_accelerations()
        times = np.array([wp.time for wp in self.waypoints])

        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Top row: Linear motion
        # Position profiles
        axes[0, 0].plot(times, positions[:, 0], "r-", label="X")
        axes[0, 0].plot(times, positions[:, 1], "g-", label="Y")
        axes[0, 0].plot(times, positions[:, 2], "b-", label="Z")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Position (m)")
        axes[0, 0].set_title("Linear Position")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Linear velocity profiles
        axes[0, 1].plot(times, velocities[:, 0], "r-", label="Vx")
        axes[0, 1].plot(times, velocities[:, 1], "g-", label="Vy")
        axes[0, 1].plot(times, velocities[:, 2], "b-", label="Vz")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Linear Velocity (m/s)")
        axes[0, 1].set_title("Linear Velocity")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Linear acceleration profiles
        axes[0, 2].plot(times, accelerations[:, 0], "r-", label="Ax")
        axes[0, 2].plot(times, accelerations[:, 1], "g-", label="Ay")
        axes[0, 2].plot(times, accelerations[:, 2], "b-", label="Az")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].set_ylabel("Linear Acceleration (m/s²)")
        axes[0, 2].set_title("Linear Acceleration")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Bottom row: Angular motion
        # Rotation profiles (Euler angles)
        euler_angles = []
        for rot in rotations:
            r = Rotation.from_matrix(rot)
            euler_angles.append(r.as_euler("xyz", degrees=True))
        euler_angles = np.array(euler_angles)

        axes[1, 0].plot(times, euler_angles[:, 0], "r-", label="Roll")
        axes[1, 0].plot(times, euler_angles[:, 1], "g-", label="Pitch")
        axes[1, 0].plot(times, euler_angles[:, 2], "b-", label="Yaw")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Rotation (degrees)")
        axes[1, 0].set_title("Angular Position")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Angular velocity profiles
        axes[1, 1].plot(times, angular_velocities[:, 0], "r-", label="ωx")
        axes[1, 1].plot(times, angular_velocities[:, 1], "g-", label="ωy")
        axes[1, 1].plot(times, angular_velocities[:, 2], "b-", label="ωz")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Angular Velocity (rad/s)")
        axes[1, 1].set_title("Angular Velocity")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Angular acceleration profiles
        axes[1, 2].plot(times, angular_accelerations[:, 0], "r-", label="αx")
        axes[1, 2].plot(times, angular_accelerations[:, 1], "g-", label="αy")
        axes[1, 2].plot(times, angular_accelerations[:, 2], "b-", label="αz")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("Angular Acceleration (rad/s²)")
        axes[1, 2].set_title("Angular Acceleration")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        return fig


class MotionPlanner:
    def __init__(self):
        pass

    def hold(
        self,
        pose: np.ndarray,        # 4x4 homogenous pose to hold
        duration: float,         # seconds to pause
        dt: float,               # timestep
        label: str = "hold"
    ) -> "Trajectory":
        """Generate a hold/pause trajectory at a fixed pose."""
        position = pose[:3, 3]
        rotation = pose[:3, :3]

        params = {
            "pose": pose.tolist(),
            "duration": duration,
            "dt": dt,
        }
        traj = Trajectory(label, params)

        num_steps = max(1, int(round(duration / dt)))
        t = 0.0
        for i in range(num_steps + 1):
            traj.add_waypoint(
                position=position.copy(),
                rotation=rotation.copy(),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
                angular_acceleration=np.zeros(3),
                index=i,
                time=t,
            )
            t += dt
        return traj

    def linear(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        dt: float,
        max_lin_vel: float = 0.5,          # NEW: max linear speed  [m/s]
        max_lin_acc: float = 0.25,          # NEW: max linear accel  [m/s^2]
        max_ang_vel: float = 0.5,          # NEW: max angular speed [rad/s]
        max_ang_acc: float = 0.25,          # NEW: max angular accel [rad/s^2]
    ) -> "Trajectory":
        """
        Linear SE(3) interpolation with SEPARATE linear and angular limits.

        If max_lin_*/max_ang_* are not given, falls back to legacy
        max_velocity / max_acceleration (same bound used for both).
        """
        start_position = start_pose[:3, 3]
        end_position   = end_pose[:3, 3]
        start_rotation = start_pose[:3, :3]
        end_rotation   = end_pose[:3, :3]

        # Distances
        linear_distance = np.linalg.norm(end_position - start_position)

        start_rot = Rotation.from_matrix(start_rotation)
        end_rot   = Rotation.from_matrix(end_rotation)
        rel_rotation     = end_rot * start_rot.inv()
        angular_distance = np.linalg.norm(rel_rotation.as_rotvec())  # [rad]

        # If still None, we treat that component as "unconstrained" and
        # will just run it at constant speed over the chosen duration.

        # ---------- small helpers: 1D trapezoidal/triangular profile ----------
        def build_profile(D, v_max, a_max):
            """
            Build minimal-time 1D profile for distance D (>=0) with
            velocity limit v_max and accel limit a_max.
            Returns (T_min, v_peak, a_used, triangular_flag).
            If v_max or a_max is None or D == 0, returns (0, 0, 0, True).
            """
            if D <= 0 or v_max is None or a_max is None or v_max <= 0 or a_max <= 0:
                return 0.0, 0.0, 0.0, True

            t_acc  = v_max / a_max
            s_acc  = 0.5 * a_max * t_acc**2

            if D <= 2 * s_acc:
                # triangular: never reach v_max
                t_acc = np.sqrt(D / a_max)
                T_min = 2 * t_acc
                v_peak = a_max * t_acc
                return T_min, v_peak, a_max, True
            else:
                # trapezoidal
                s_const = D - 2 * s_acc
                t_const = s_const / v_max
                T_min = 2 * t_acc + t_const
                v_peak = v_max
                return T_min, v_peak, a_max, False

        def eval_profile(t, D, v_max, a_max, T_min, v_peak, a_used, triangular):
            """
            Evaluate s(t), s_dot(t), s_ddot(t) for a given 1D profile.
            If v_max/a_max are None or D==0 or T_min==0 => stays at 0.
            """
            if D <= 0 or v_max is None or a_max is None or v_max <= 0 or a_max <= 0 or T_min <= 0:
                return 0.0, 0.0, 0.0

            # Clamp t to [0, T_min]
            if t <= 0.0:
                return 0.0, 0.0, 0.0
            if t >= T_min:
                return D, 0.0, 0.0

            if triangular:
                t_acc = T_min * 0.5
                if t <= t_acc:
                    # accel phase
                    s      = 0.5 * a_used * t**2
                    s_dot  = a_used * t
                    s_ddot = a_used
                else:
                    # decel phase (mirror)
                    td     = T_min - t
                    s      = D - 0.5 * a_used * td**2
                    s_dot  = a_used * td
                    s_ddot = -a_used
                return s, s_dot, s_ddot
            else:
                # trapezoid: accel -> const -> decel
                t_acc = v_peak / a_used
                s_acc = 0.5 * a_used * t_acc**2
                t_const = T_min - 2 * t_acc

                if t <= t_acc:
                    s      = 0.5 * a_used * t**2
                    s_dot  = a_used * t
                    s_ddot = a_used
                elif t <= t_acc + t_const:
                    s      = s_acc + v_peak * (t - t_acc)
                    s_dot  = v_peak
                    s_ddot = 0.0
                else:
                    td     = T_min - t
                    s      = D - 0.5 * a_used * td**2
                    s_dot  = a_used * td
                    s_ddot = -a_used
                return s, s_dot, s_ddot

        # ---------- build separate linear & angular profiles ----------
        T_lin_min, v_lin_peak, a_lin_used, tri_lin = build_profile(
            linear_distance, max_lin_vel, max_lin_acc
        )
        T_ang_min, v_ang_peak, a_ang_used, tri_ang = build_profile(
            angular_distance, max_ang_vel, max_ang_acc
        )

        # Global motion duration: big enough for both
        T_total = max(T_lin_min, T_ang_min, 0.0)
        if T_total <= 0:
            # Degenerate: no motion
            trajectory_time = 0.0
        else:
            trajectory_time = T_total

        # Fallback: if everything degenerate, match previous logic (one step)
        if trajectory_time == 0.0:
            num_steps = 1
        else:
            num_steps = max(1, int(np.round(trajectory_time / dt)))

        params = {
            "start_position": start_position.tolist(),
            "end_position": end_position.tolist(),
            "start_rotation": start_rotation.tolist(),
            "end_rotation": end_rotation.tolist(),
            "time": trajectory_time,
            "dt": dt,
            "linear_distance": linear_distance,
            "angular_distance": angular_distance,
            "max_lin_vel": max_lin_vel,
            "max_lin_acc": max_lin_acc,
            "max_ang_vel": max_ang_vel,
            "max_ang_acc": max_ang_acc,
        }

        trajectory = Trajectory("linear", params)

        # Slerp for orientation; we'll drive it with an "alpha_ang" in [0,1]
        slerp = Slerp([0, 1], Rotation.from_matrix([start_rotation, end_rotation]))

        for i in range(num_steps + 1):
            if trajectory_time == 0.0:
                t = 0.0
            else:
                t = min(i * dt, trajectory_time)

            # Linear profile (on arc length D_lin)
            s_lin, sdot_lin, sddot_lin = eval_profile(
                t, linear_distance, max_lin_vel, max_lin_acc,
                T_lin_min, v_lin_peak, a_lin_used, tri_lin
            )
            if linear_distance > 1e-9:
                alpha_lin = s_lin / linear_distance
                alpha_lin = max(0.0, min(1.0, alpha_lin))
                linear_direction = (end_position - start_position) / linear_distance
            else:
                alpha_lin = 0.0
                linear_direction = np.zeros(3)

            position = start_position + alpha_lin * (end_position - start_position)
            linear_velocity = linear_direction * sdot_lin
            linear_acceleration = linear_direction * sddot_lin

            # Angular profile (on angle D_ang)
            s_ang, sdot_ang, sddot_ang = eval_profile(
                t, angular_distance, max_ang_vel, max_ang_acc,
                T_ang_min, v_ang_peak, a_ang_used, tri_ang
            )
            if angular_distance > 1e-9:
                alpha_ang = s_ang / angular_distance
                alpha_ang = max(0.0, min(1.0, alpha_ang))
                rot_axis = rel_rotation.as_rotvec() / angular_distance
            else:
                alpha_ang = 0.0
                rot_axis = np.zeros(3)

            # Orientation from slerp driven by angular progress
            rotation = slerp(alpha_ang).as_matrix()

            # Angular velocity/acceleration in body/world frame
            angular_velocity = rot_axis * sdot_ang
            angular_acceleration = rot_axis * sddot_ang

            trajectory.add_waypoint(
                position=position,
                rotation=rotation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                angular_acceleration=angular_acceleration,
                index=i,
                time=t,
            )

        return trajectory

    
    def circular(
        self,
        start_pose: np.ndarray,
        dt: float,
        radius: float = None,
        diameter: float = None,
        max_velocity: float = None,      # along arc, m/s
        max_acceleration: float = None,  # along arc, m/s^2
        keep_orientation: bool = True,   # keep start orientation; if False, yaw-tangent
        ccw: bool = True,                # CCW by default
        label: str = "circular"
    ) -> "Trajectory":
        """
        Generate a full 360° circular trajectory in the XY plane at start_pose.z.
        Center is at start_position - [radius, 0, 0] (world X), so we begin on the circle at theta=0.
        CCW means +theta over time.

        Timing uses trapezoidal profile on arc-length 's' with v_max/a_max (like 'linear').
        """

        # Resolve radius
        if radius is None and diameter is None:
            raise ValueError("Provide either radius or diameter")
        if radius is None:
            radius = diameter * 0.5
        if radius <= 0:
            raise ValueError("radius must be > 0")

        # Extract start position/orientation
        p0 = start_pose[:3, 3].copy()
        R0 = start_pose[:3, :3].copy()

        # Circle center: start_pose - radius along world +X
        center = p0.copy()
        center[0] -= radius  # world-frame –x shift

        # Geometry & travel
        s_total = 2.0 * np.pi * radius  # arc length for 360°
        if max_velocity and max_acceleration and max_velocity > 0 and max_acceleration > 0:
            # Trapezoidal timing on arc length
            t_accel = max_velocity / max_acceleration
            s_accel = 0.5 * max_acceleration * t_accel**2

            if s_total <= 2 * s_accel:
                # Triangular
                t_accel = np.sqrt(s_total / max_acceleration)
                T = 2 * t_accel
                v_peak = max_acceleration * t_accel
            else:
                s_const = s_total - 2 * s_accel
                t_const = s_const / max_velocity
                T = 2 * t_accel + t_const
                v_peak = max_velocity
        elif max_velocity and max_velocity > 0:
            # Constant speed around the circle
            T = s_total / max_velocity
            v_peak = max_velocity
            max_acceleration = None
        else:
            # Fallback: discretize by dt, keep a reasonable speed
            # (match your 'linear' fallback style)
            num_steps = max(1, int(s_total / dt / 1.0))  # ~1 m/s nominal
            T = num_steps * dt
            v_peak = s_total / T if T > 0 else 0

        params = {
            "type": "circle",
            "center": center.tolist(),
            "radius": radius,
            "dt": dt,
            "time": T,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "keep_orientation": keep_orientation,
            "ccw": ccw,
        }
        traj = Trajectory(label, params)

        num_steps = int(T / dt)
        # Direction sign for CCW/CW
        dir_sign = 1.0 if ccw else -1.0

        # Helper to compute s, s_dot, s_ddot from trapezoid
        def arc_profile(t: float):
            if max_acceleration and v_peak and T:
                t_acc = v_peak / max_acceleration
                s_acc = 0.5 * max_acceleration * t_acc**2

                if T <= 2 * t_acc + 1e-9:
                    # Triangular
                    t_acc = T * 0.5
                    if t <= t_acc:
                        s = 0.5 * max_acceleration * t**2
                        s_dot = max_acceleration * t
                        s_ddot = max_acceleration
                    else:
                        td = T - t
                        s = s_total - 0.5 * max_acceleration * td**2
                        s_dot = max_acceleration * td
                        s_ddot = -max_acceleration
                else:
                    t_const = T - 2 * t_acc
                    if t <= t_acc:
                        s = 0.5 * max_acceleration * t**2
                        s_dot = max_acceleration * t
                        s_ddot = max_acceleration
                    elif t <= t_acc + t_const:
                        s = s_acc + v_peak * (t - t_acc)
                        s_dot = v_peak
                        s_ddot = 0.0
                    else:
                        td = T - t
                        s = s_total - 0.5 * max_acceleration * td**2
                        s_dot = max_acceleration * td
                        s_ddot = -max_acceleration
            else:
                # Constant speed
                s = (s_total / T) * t if T > 0 else 0.0
                s_dot = s_total / T if T > 0 else 0.0
                s_ddot = 0.0
            return s, s_dot, s_ddot

        # Generate waypoints
        for i in range(num_steps + 1):
            t = min(i * dt, T)
            s, s_dot, s_ddot = arc_profile(t)

            # Angle & its derivatives
            theta = dir_sign * (s / radius)          # rad
            theta_dot = dir_sign * (s_dot / radius)  # rad/s
            theta_ddot = dir_sign * (s_ddot / radius)# rad/s^2

            # Parametric circle (start at theta=0 at start point p0)
            c, s_ = np.cos(theta), np.sin(theta)
            x = center[0] + radius * (1.0 * c)   # starts at x = center_x + r = p0_x
            y = center[1] + radius * (1.0 * s_)  # starts at y = center_y = p0_y
            z = p0[2]                             # plane at start z

            position = np.array([x, y, z])

            # Linear velocity & acceleration in world (XY plane)
            # r*[ -sin, cos ] * theta_dot ;  r*[ -cos, -sin ]*theta_dot^2 + r*[ -sin, cos ]*theta_ddot
            vx = -radius * s_ * theta_dot
            vy =  radius * c  * theta_dot
            ax = -radius * c  * (theta_dot**2) - radius * s_ * theta_ddot
            ay = -radius * s_ * (theta_dot**2) + radius * c  * theta_ddot

            linear_velocity = np.array([vx, vy, 0.0])
            linear_acceleration = np.array([ax, ay, 0.0])

            # Orientation handling
            if keep_orientation:
                R = R0
            else:
                # Align yaw with tangent direction (tool x-axis along velocity)
                # Tangent vector (vx, vy) -> yaw = atan2(vy, vx)
                # If speed ~0 (at very start/end), fall back to initial yaw.
                if np.hypot(vx, vy) > 1e-6:
                    yaw = np.arctan2(vy, vx)
                else:
                    yaw = 0.0
                # Build rotation = yaw about world Z times original roll/pitch (extract them from R0)
                # Simple approach: take R0’s z-axis to keep tool’s z aligned with world z, rotate about z.
                Rz = Rotation.from_euler("z", yaw).as_matrix()
                # Keep roll/pitch from R0 by projecting its z-axis, or for simplicity just use Rz
                # (If you want full roll/pitch preservation, replace this block with a decomposition.)
                R = Rz @ np.eye(3)

            # We don't command angular velocity/accel here; keep zero or compute if you align yaw.
            angular_velocity = np.zeros(3)
            angular_acceleration = np.zeros(3)

            traj.add_waypoint(
                position=position,
                rotation=R,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                angular_acceleration=angular_acceleration,
                index=i,
                time=t,
            )

        return traj

    def circular_xz(
        self,
        start_pose: np.ndarray,
        dt: float,
        radius: float = None,
        diameter: float = None,
        max_velocity: float = None,      # along arc, m/s
        max_acceleration: float = None,  # along arc, m/s^2
        keep_orientation: bool = True,
        ccw: bool = True,
        label: str = "circular_xz"
    ) -> "Trajectory":
        """
        Generate a full 360° circular trajectory in the XZ plane at start_pose.y.
        Center is at start_position - [radius, 0, 0] (world X), so we begin
        on the circle at theta=0.

        Timing uses trapezoidal profile on arc-length 's' with v_max/a_max.
        """
        # Resolve radius
        if radius is None and diameter is None:
            raise ValueError("Provide either radius or diameter")
        if radius is None:
            radius = diameter * 0.5
        if radius <= 0:
            raise ValueError("radius must be > 0")

        # Extract start position/orientation
        p0 = start_pose[:3, 3].copy()
        R0 = start_pose[:3, :3].copy()

        # Circle center: shift along world +X, same Y, same Z-center
        center = p0.copy()
        center[0] -= radius  # x center
        # center[1] = p0[1]   # y fixed
        # center[2] = p0[2]   # z center

        # Geometry & travel
        s_total = 2.0 * np.pi * radius  # arc length for 360°
        if max_velocity and max_acceleration and max_velocity > 0 and max_acceleration > 0:
            t_accel = max_velocity / max_acceleration
            s_accel = 0.5 * max_acceleration * t_accel**2

            if s_total <= 2 * s_accel:
                # Triangular profile
                t_accel = np.sqrt(s_total / max_acceleration)
                T = 2 * t_accel
                v_peak = max_acceleration * t_accel
            else:
                s_const = s_total - 2 * s_accel
                t_const = s_const / max_velocity
                T = 2 * t_accel + t_const
                v_peak = max_velocity
        elif max_velocity and max_velocity > 0:
            # Constant speed
            T = s_total / max_velocity
            v_peak = max_velocity
            max_acceleration = None
        else:
            # Fallback
            num_steps = max(1, int(s_total / dt / 1.0))
            T = num_steps * dt
            v_peak = s_total / T if T > 0 else 0.0

        params = {
            "type": "circle_xz",
            "center": center.tolist(),
            "radius": radius,
            "dt": dt,
            "time": T,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "keep_orientation": keep_orientation,
            "ccw": ccw,
        }
        traj = Trajectory(label, params)

        num_steps = int(T / dt)
        dir_sign = 1.0 if ccw else -1.0

        def arc_profile(t: float):
            if max_acceleration and v_peak and T:
                t_acc = v_peak / max_acceleration
                s_acc = 0.5 * max_acceleration * t_acc**2

                if T <= 2 * t_acc + 1e-9:
                    # Triangular
                    t_acc = T * 0.5
                    if t <= t_acc:
                        s = 0.5 * max_acceleration * t**2
                        s_dot = max_acceleration * t
                        s_ddot = max_acceleration
                    else:
                        td = T - t
                        s = s_total - 0.5 * max_acceleration * td**2
                        s_dot = max_acceleration * td
                        s_ddot = -max_acceleration
                else:
                    t_const = T - 2 * t_acc
                    if t <= t_acc:
                        s = 0.5 * max_acceleration * t**2
                        s_dot = max_acceleration * t
                        s_ddot = max_acceleration
                    elif t <= t_acc + t_const:
                        s = s_acc + v_peak * (t - t_acc)
                        s_dot = v_peak
                        s_ddot = 0.0
                    else:
                        td = T - t
                        s = s_total - 0.5 * max_acceleration * td**2
                        s_dot = max_acceleration * td
                        s_ddot = -max_acceleration
            else:
                # Constant speed
                s = (s_total / T) * t if T > 0 else 0.0
                s_dot = s_total / T if T > 0 else 0.0
                s_ddot = 0.0
            return s, s_dot, s_ddot

        for i in range(num_steps + 1):
            t = min(i * dt, T)
            s, s_dot, s_ddot = arc_profile(t)

            theta = dir_sign * (s / radius)
            theta_dot = dir_sign * (s_dot / radius)
            theta_ddot = dir_sign * (s_ddot / radius)

            c = np.cos(theta)
            s_ = np.sin(theta)

            # XZ circle at fixed y = p0[1]
            x = center[0] + radius * c
            y = p0[1]
            z = center[2] + radius * s_
            position = np.array([x, y, z])

            # Vel/acc in XZ plane
            vx = -radius * s_ * theta_dot
            vz =  radius * c  * theta_dot
            ax = -radius * c  * (theta_dot**2) - radius * s_ * theta_ddot
            az = -radius * s_ * (theta_dot**2) + radius * c  * theta_ddot

            linear_velocity = np.array([vx, 0.0, vz])
            linear_acceleration = np.array([ax, 0.0, az])

            if keep_orientation:
                R = R0
            else:
                # For now we won't use this branch (no rotations), but keep it for later.
                R = R0

            angular_velocity = np.zeros(3)
            angular_acceleration = np.zeros(3)

            traj.add_waypoint(
                position=position,
                rotation=R,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                angular_acceleration=angular_acceleration,
                index=i,
                time=t,
            )

        return traj
    
    def concatenate_trajectories(self, trajectories: List[Trajectory]) -> Trajectory:
        """Concatenate multiple trajectories into one continuous trajectory"""
        if not trajectories:
            raise ValueError("No trajectories to concatenate")

        combined_params = {
            "concatenated_from": [traj.primitive_type for traj in trajectories],
            "num_segments": len(trajectories),
        }

        concatenated = Trajectory("concatenated", combined_params)

        time_offset = 0.0
        waypoint_index = 0

        for traj_idx, trajectory in enumerate(trajectories):
            for wp_idx, waypoint in enumerate(trajectory.waypoints):
                if traj_idx > 0 and wp_idx == 0:
                    continue

                new_time = waypoint.time + time_offset
                concatenated.add_waypoint(
                    waypoint.position,
                    waypoint.rotation,
                    waypoint.linear_velocity,
                    waypoint.angular_velocity,
                    waypoint.linear_acceleration,
                    waypoint.angular_acceleration,
                    waypoint_index,
                    new_time,
                )
                waypoint_index += 1

            if trajectory.waypoints:
                time_offset += trajectory.waypoints[-1].time

        return concatenated


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
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )
    pose3 = np.array(
        [
            [1, 0, 0, -1],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )
    pose4 = np.array(
        [
            [1, 0, 0, -1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # 90 around z axis
    pose5 = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    pose6 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0.25],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )
    pose7 = np.array(        [
            [1, 0, 0, -0.25],
            [0, 1, 0, 0.25],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )
    pose8 = np.array(        [
            [1, 0, 0, -0.25],
            [0, 1, 0, 0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )
    pose9 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.4],
            [0, 0, 0, 1],
        ]
    )

    hz = 100
    dt = 1 / hz
    max_lin_vel = 0.5 
    max_lin_acc = 0.25
    max_ang_vel = 0.5
    max_ang_acc = 0.25

    theta = np.deg2rad(30.0)
    R_y = np.array([
        [ np.cos(theta),  0.0, np.sin(theta)],
        [ 0.0,            1.0, 0.0          ],
        [-np.sin(theta),  0.0, np.cos(theta)],
    ])

    pose_twist_y = pose2.copy()
    # rotate relative to pose2's current orientation (in case pose2 ever changes)
    pose_twist_y[:3, :3] = pose2[:3, :3] @ R_y
    twist_y = planner.linear(
        pose2,
        pose_twist_y,
        dt
    )

    up = planner.linear(pose1, pose2, dt, max_lin_vel = max_lin_vel, max_lin_acc = max_lin_acc, max_ang_vel = max_ang_vel,max_ang_acc = max_ang_acc)
    # UP = planner.linear(pose1, pose9, dt, max_velocity=0.25, max_acceleration=0.1)
    # side = planner.linear(pose2, pose3, dt, max_velocity=0.5, max_acceleration=0.25)
    # down = planner.linear(pose3, pose4, dt, max_velocity=0.5, max_acceleration=0.25)
    # twist = planner.linear(pose2, pose5, dt, max_lin_vel = max_lin_vel, max_lin_acc = max_lin_acc, max_ang_vel = max_ang_vel,max_ang_acc = max_ang_acc)
    # side_y = planner.linear(pose2, pose6, dt, max_velocity=0.25, max_acceleration=0.2)
    # side_y_back = planner.linear(
    #     pose6, pose2, dt, max_velocity=0.25, max_acceleration=0.2
    # )
    # hold_side = planner.hold(pose6, 1.0, dt)
    # hold_middle = planner.hold(pose2, 1.0, dt)

    # hold_t = 1
    # hold_0 = planner.hold(pose2, hold_t, dt)
    # side_y = planner.linear(pose2, pose6, dt, max_velocity=0.15, max_acceleration=0.15)
    # hold_1 = planner.hold(pose6, hold_t, dt)
    # side_x = planner.linear(pose6, pose7, dt, max_velocity=0.15, max_acceleration=0.15)
    # hold_2 = planner.hold(pose7, hold_t, dt)
    # side_y_b = planner.linear(pose7, pose8, dt, max_velocity=0.15, max_acceleration=0.15)
    # hold_3 = planner.hold(pose8, hold_t, dt)
    # side_x_b = planner.linear(pose8, pose2, dt, max_velocity=0.15, max_acceleration=0.15)
    # down = planner.linear(pose2, pose1, dt, max_velocity=0.15, max_acceleration=0.15)
    # hold_9 = planner.hold(pose9, hold_t, dt)

    
    # circle = planner.circular(
    #     start_pose=pose2,
    #     dt=dt,
    #     radius=0.2,
    #     max_velocity=0.25,        # m/s along arc
    #     max_acceleration=0.2,    # m/s^2 along arc
    #     keep_orientation=True,     # keep same tool orientation
    #     ccw=True,
    #     label="circle_xy_ccw"
    # )


    # full_trajectory = planner.concatenate_trajectories([up, hold_middle, side_y, hold_side, side_y_back, hold_middle, side_y, hold_side, side_y_back]) #side_y
    # full_trajectory = planner.concatenate_trajectories([up, hold_0, side_y, hold_1, side_x, hold_2, side_y_b, hold_3, side_x_b, hold_0, down]) #side_y #rectangle
    # full_trajectory = planner.concatenate_trajectories([up, circle, down]) #circle
    full_trajectory = planner.concatenate_trajectories([up, twist_y]) #twist
    #full_trajectory = planner.concatenate_trajectories([UP, hold_9])
    fig3d, _ = full_trajectory.plot_3d()
    fig3d.savefig("plots/trajectory_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = full_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_profiles.png", dpi=150, bbox_inches="tight")

    # Ensure directory exists
    import os
    os.makedirs("plots", exist_ok=True)

    full_trajectory.save_trajectory("motion_planner/trajectories_old/twist_y.npz")

