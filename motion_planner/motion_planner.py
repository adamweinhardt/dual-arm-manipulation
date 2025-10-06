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
        max_velocity: float = None,
        max_acceleration: float = None,
    ) -> Trajectory:
        start_position = start_pose[:3, 3]
        end_position = end_pose[:3, 3]
        start_rotation = start_pose[:3, :3]
        end_rotation = end_pose[:3, :3]

        # Calculate path lengths
        linear_distance = np.linalg.norm(end_position - start_position)

        start_rot = Rotation.from_matrix(start_rotation)
        end_rot = Rotation.from_matrix(end_rotation)
        rel_rotation = end_rot * start_rot.inv()
        angular_distance = np.linalg.norm(rel_rotation.as_rotvec())

        dominant_distance = max(linear_distance, angular_distance)

        if max_acceleration and max_velocity:
            # Trapezoidal velocity profile with acceleration limits
            # Time to reach max velocity: t_accel = v_max / a_max
            # Distance during acceleration: s_accel = 0.5 * a_max * t_accel^2
            t_accel = max_velocity / max_acceleration
            s_accel = 0.5 * max_acceleration * t_accel**2

            if dominant_distance <= 2 * s_accel:
                # Triangular profile (never reach max velocity)
                # s = 0.5 * a * t^2, so t = sqrt(2*s/a)
                t_accel = np.sqrt(dominant_distance / max_acceleration)
                trajectory_time = 2 * t_accel
                actual_max_velocity = max_acceleration * t_accel
            else:
                # Trapezoidal profile (reach max velocity)
                s_constant = dominant_distance - 2 * s_accel
                t_constant = s_constant / max_velocity
                trajectory_time = 2 * t_accel + t_constant
                actual_max_velocity = max_velocity

        elif max_velocity:
            # Constant velocity (no acceleration limit)
            trajectory_time = dominant_distance / max_velocity
            actual_max_velocity = max_velocity

        else:
            # Original method: distance-based discretization
            num_steps = max(1, int(dominant_distance / dt))
            trajectory_time = num_steps * dt
            actual_max_velocity = (
                dominant_distance / trajectory_time if trajectory_time > 0 else 0
            )

        params = {
            "start_position": start_position.tolist(),
            "end_position": end_position.tolist(),
            "start_rotation": start_rotation.tolist(),
            "end_rotation": end_rotation.tolist(),
            "time": trajectory_time,
            "dt": dt,
            "linear_distance": linear_distance,
            "angular_distance": angular_distance,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
        }

        trajectory = Trajectory("linear", params)
        num_steps = int(trajectory_time / dt)
        slerp = Slerp([0, 1], Rotation.from_matrix([start_rotation, end_rotation]))

        # Generate waypoints with proper velocity/acceleration profiles
        for i in range(num_steps + 1):
            t = i * dt

            # Calculate motion profile parameters
            if max_acceleration and max_velocity:
                t_accel = (
                    max_velocity / max_acceleration
                    if dominant_distance
                    > 2
                    * (0.5 * max_acceleration * (max_velocity / max_acceleration) ** 2)
                    else np.sqrt(dominant_distance / max_acceleration)
                )

                if t <= t_accel:
                    # Acceleration phase
                    alpha = 0.5 * max_acceleration * t**2 / dominant_distance
                    velocity_magnitude = max_acceleration * t
                    acceleration_magnitude = max_acceleration
                elif t <= trajectory_time - t_accel:
                    # Constant velocity phase
                    s_accel = 0.5 * max_acceleration * t_accel**2
                    s_const = actual_max_velocity * (t - t_accel)
                    alpha = (s_accel + s_const) / dominant_distance
                    velocity_magnitude = actual_max_velocity
                    acceleration_magnitude = 0
                else:
                    # Deceleration phase
                    t_decel = trajectory_time - t
                    s_accel = 0.5 * max_acceleration * t_accel**2
                    s_const = (
                        actual_max_velocity * (trajectory_time - 2 * t_accel)
                        if trajectory_time > 2 * t_accel
                        else 0
                    )
                    # Fix: calculate distance traveled during deceleration from the END
                    s_decel = 0.5 * max_acceleration * t_decel**2
                    alpha = (dominant_distance - s_decel) / dominant_distance
                    velocity_magnitude = max_acceleration * t_decel
                    acceleration_magnitude = -max_acceleration
            else:
                # Simple linear interpolation
                alpha = t / trajectory_time if trajectory_time > 0 else 0
                velocity_magnitude = (
                    dominant_distance / trajectory_time if trajectory_time > 0 else 0
                )
                acceleration_magnitude = 0

            # Clamp alpha to [0, 1]
            alpha = max(0, min(1, alpha))

            # Calculate position and rotation
            position = start_position + alpha * (end_position - start_position)
            rotation = slerp(alpha).as_matrix()

            # Calculate velocity and acceleration vectors
            if linear_distance > 0:
                linear_direction = (end_position - start_position) / linear_distance
                linear_velocity = (
                    linear_direction
                    * velocity_magnitude
                    * (linear_distance / dominant_distance)
                )
                linear_acceleration = (
                    linear_direction
                    * acceleration_magnitude
                    * (linear_distance / dominant_distance)
                )
            else:
                linear_velocity = np.zeros(3)
                linear_acceleration = np.zeros(3)

            if angular_distance > 0:
                angular_direction = rel_rotation.as_rotvec() / angular_distance
                angular_velocity = (
                    angular_direction
                    * velocity_magnitude
                    * (angular_distance / dominant_distance)
                )
                angular_acceleration = (
                    angular_direction
                    * acceleration_magnitude
                    * (angular_distance / dominant_distance)
                )
            else:
                angular_velocity = np.zeros(3)
                angular_acceleration = np.zeros(3)

            trajectory.add_waypoint(
                position,
                rotation,
                linear_velocity,
                angular_velocity,
                linear_acceleration,
                angular_acceleration,
                i,
                t,
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

    hz = 100
    dt = 1 / hz
    up = planner.linear(pose1, pose2, dt, max_velocity=0.5, max_acceleration=0.25)
    side = planner.linear(pose2, pose3, dt, max_velocity=0.5, max_acceleration=0.25)
    down = planner.linear(pose3, pose4, dt, max_velocity=0.5, max_acceleration=0.25)
    twist = planner.linear(pose2, pose5, dt, max_velocity=0.25, max_acceleration=0.1)
    side_y = planner.linear(pose2, pose6, dt, max_velocity=0.05, max_acceleration=0.1)
    side_y_back = planner.linear(
        pose6, pose2, dt, max_velocity=0.05, max_acceleration=0.1
    )
    hold_side = planner.hold(pose6, 1.0, dt)
    hold_middle = planner.hold(pose2, 1.0, dt)

    hold_t = 0.5
    hold_0 = planner.hold(pose2, hold_t, dt)
    side_y = planner.linear(pose2, pose6, dt, max_velocity=0.15, max_acceleration=0.15)
    hold_1 = planner.hold(pose6, hold_t, dt)
    side_x = planner.linear(pose6, pose7, dt, max_velocity=0.15, max_acceleration=0.15)
    hold_2 = planner.hold(pose7, hold_t, dt)
    side_y_b = planner.linear(pose7, pose8, dt, max_velocity=0.15, max_acceleration=0.15)
    hold_3 = planner.hold(pose8, hold_t, dt)
    side_x_b = planner.linear(pose8, pose2, dt, max_velocity=0.15, max_acceleration=0.15)
    down = planner.linear(pose2, pose1, dt, max_velocity=0.15, max_acceleration=0.15)

    
    circle = planner.circular(
        start_pose=pose2,
        dt=dt,
        radius=0.2,
        max_velocity=0.2,        # m/s along arc
        max_acceleration=0.1,    # m/s^2 along arc
        keep_orientation=True,     # keep same tool orientation
        ccw=True,
        label="circle_xy_ccw"
    )




    # full_trajectory = planner.concatenate_trajectories([up, hold_middle, side_y, hold_side, side_y_back, hold_middle, side_y, hold_side, side_y_back]) #side_y
    # full_trajectory = planner.concatenate_trajectories([up, hold_0, side_y, hold_1, side_x, hold_2, side_y_b, hold_3, side_x_b, hold_0, down]) #side_y #rectangle
    # full_trajectory = planner.concatenate_trajectories([up, circle, down]) #circle
    full_trajectory = planner.concatenate_trajectories([up, twist]) #twist
    fig3d, _ = full_trajectory.plot_3d()
    fig3d.savefig("plots/trajectory_3d.png", dpi=150, bbox_inches="tight")

    fig_profiles = full_trajectory.plot_profiles()
    fig_profiles.savefig("plots/trajectory_profiles.png", dpi=150, bbox_inches="tight")

    # Ensure directory exists
    import os
    os.makedirs("plots", exist_ok=True)

    full_trajectory.save_trajectory("motion_planner/trajectories/twist.npz")
