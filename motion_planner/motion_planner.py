import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
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
            [0, 1, 0, 0.4],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1],
        ]
    )

    hz = 100
    dt = 1 / hz
    up = planner.linear(pose1, pose2, dt, max_velocity=0.5, max_acceleration=0.25)
    side = planner.linear(pose2, pose3, dt, max_velocity=0.5, max_acceleration=0.25)
    down = planner.linear(pose3, pose4, dt, max_velocity=0.5, max_acceleration=0.25)
    twist = planner.linear(pose2, pose5, dt, max_velocity=0.5, max_acceleration=0.25)
    side_y = planner.linear(pose2, pose6, dt, max_velocity=0.3, max_acceleration=0.25)
    side_y_back = planner.linear(
        pose6, pose2, dt, max_velocity=0.3, max_acceleration=0.25
    )

    full_trajectory = planner.concatenate_trajectories(
        [up, side_y, side_y_back, side_y, side_y_back]
    )
    full_trajectory.plot_3d()
    full_trajectory.plot_profiles()
    plt.show()

    full_trajectory.save_trajectory("motion_planner/trajectories/side_y.npz")
