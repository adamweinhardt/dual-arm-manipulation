import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def plot_trajectory_heatmap(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    plt.rcParams.update({"font.size": 14})
    LABEL_SIZE = 24
    TITLE_SIZE = 18
    TRAJECTORY_THICKNESS = 120

    try:
        data = np.load(file_path, allow_pickle=True)
        if "position" not in data or "linear_velocity" not in data:
            print(
                "❌ Error: .npz file must contain 'position' and 'linear_velocity' arrays."
            )
            return

        positions = data["position"]
        velocities = data["linear_velocity"]

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    speed = np.linalg.norm(velocities, axis=1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    scatter = ax.scatter(
        x,
        y,
        z,
        c=speed,
        cmap="turbo",
        s=TRAJECTORY_THICKNESS,
        alpha=0.9,
        edgecolor="none",
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(
        "Absolute Linear Velocity (m/s)", rotation=270, labelpad=25, fontsize=LABEL_SIZE
    )
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel("X Position (m)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_ylabel("Y Position (m)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_zlabel("Z Position (m)", fontsize=LABEL_SIZE, labelpad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)

    try:
        max_range = (
            np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            / 2.0
        )
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except:
        pass

    plt.show()


if __name__ == "__main__":
    target_file = "motion_planner/trajectories/rollercoaster_circle.npz"
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    plot_trajectory_heatmap(target_file)
