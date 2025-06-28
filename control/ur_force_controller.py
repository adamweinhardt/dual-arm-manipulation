import time
import threading
import numpy as np
from ur_controller import URController


class PIDController:
    """Basic PID controller for force control"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=0.5, dt=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.dt = dt

        # states
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = self.dt

        p_term = self.kp * error

        self.integral += error * dt
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        output = np.clip(output, -self.max_output, self.max_output)

        self.prev_error = error
        self.last_time = current_time

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class URForceController(URController):
    """URController with force control capabilities"""

    def __init__(self, ip, hz=50, kp=0.01, ki=0.0, kd=0):
        super().__init__(ip)

        self.force_control_active = False
        self.force_control_thread = None
        self.force_control_stop = threading.Event()

        self.force_pid = PIDController(kp=kp, ki=ki, kd=kd)

        self.control_rate_hz = hz
        self.min_force_threshold = 0.5

        self.force_data = []

    def force_control_to_target(
        self, reference_force, direction, distance_cap=0.2, timeout=30.0
    ):
        if self.force_control_active:
            print("Force control already active!")
            return False

        direction = np.array(direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            print("Error: Direction vector cannot be zero!")
            return False
        direction = direction / direction_norm

        # Store control parameters
        self.ref_force = reference_force
        self.control_direction = direction
        self.distance_cap = distance_cap
        self.start_position = np.array(self.get_state()["pose"][:3])
        self.start_time = time.time()
        self.control_timeout = timeout

        # Reset PID controller
        self.force_pid.reset()

        # Clear previous force data
        self.force_data = []

        # Start force control thread
        self.force_control_active = True
        self.force_control_stop.clear()
        self.force_control_thread = threading.Thread(
            target=self._force_control_loop, daemon=True
        )
        self.force_control_thread.start()

        print(f"Force control started: {reference_force}N in direction {direction}")
        print(f"Distance cap: {distance_cap}m, Timeout: {timeout}s")

        return True

    def stop_force_control(self):
        """Stop the force control loop"""
        if self.force_control_active:
            self.force_control_active = False
            self.force_control_stop.set()
            if self.force_control_thread:
                self.force_control_thread.join(timeout=2.0)
            print("Force control stopped")

    def _force_control_loop(self):
        """Main force control loop - runs in separate thread"""
        control_period = 1.0 / self.control_rate_hz

        while self.force_control_active and not self.force_control_stop.is_set():
            loop_start = time.time()

            try:
                current_state = self.get_state()

                if (
                    not current_state
                    or current_state["force"] is None
                    or current_state["pose"] is None
                ):
                    print("ERROR: Invalid robot state received, stopping force control")
                    break

                current_force_vector = np.array(current_state["force"][:3])
                current_position = np.array(current_state["pose"][:3])

                force_data_point = {
                    "timestamp": time.time() - self.start_time,
                    "force_vector": current_force_vector.copy(),
                    "force_magnitude": np.linalg.norm(current_force_vector),
                    "position": current_position.copy(),
                    "reference_force": self.ref_force,
                    "distance_from_start": np.linalg.norm(
                        current_position - self.start_position
                    ),
                }
                self.force_data.append(force_data_point)

                force_in_direction = np.dot(
                    current_force_vector, self.control_direction
                )
                force_data_point["force_in_direction"] = force_in_direction

                force_error = self.ref_force - abs(force_in_direction)

                distance_moved = np.linalg.norm(current_position - self.start_position)
                if distance_moved >= self.distance_cap:
                    print(f"Distance cap reached: {distance_moved:.3f}m")
                    break

                if time.time() - self.start_time >= self.control_timeout:
                    print(f"Force control timeout reached")
                    break

                if abs(force_error) > self.min_force_threshold:
                    pid_output = self.force_pid.update(force_error)

                    velocity = self.control_direction * pid_output

                    speed_command = [velocity[0], velocity[1], velocity[2], 0, 0, 0]
                    self.speedL(speed_command, acceleration=0.5, time_duration=0.1)

                    if int(time.time() * 10) % 10 == 0:  # Print every 1 second
                        print(
                            f"Force: {force_in_direction:.2f}N (target: {self.ref_force}N), "
                            f"Error: {force_error:.2f}N, "
                            f"Distance: {distance_moved:.3f}m"
                        )
                else:
                    self.speedStop()
                    print(
                        f"Target force achieved: {force_in_direction:.2f}N (target: {self.ref_force}N)"
                    )
                    break

            except Exception as e:
                print(f"Force control error: {e}")
                # Try to stop robot movement
                try:
                    self.speedStop()
                except:
                    print("Failed to send stop command - connection likely lost")
                break

            elapsed = time.time() - loop_start
            sleep_time = max(0, control_period - elapsed)
            time.sleep(sleep_time)

        try:
            self.speedStop()
        except:
            pass

        self.force_control_active = False
        print("Force control loop ended")

    def wait_for_force_control(self):
        """Wait for force control to complete"""
        if self.force_control_thread and self.force_control_thread.is_alive():
            self.force_control_thread.join()

    def get_current_force(self):
        """Get current TCP force reading"""
        return self.get_state()["force"]

    def get_force_magnitude(self):
        """Get magnitude of current force vector"""
        force_vector = np.array(self.get_state()["force"][:3])
        return np.linalg.norm(force_vector)

    def plot_force_data(self):
        """Generate plots for force control data"""
        if not self.force_data:
            print("No force data to plot")
            return

        import matplotlib.pyplot as plt

        timestamps = [d["timestamp"] for d in self.force_data]
        force_vectors = np.array([d["force_vector"] for d in self.force_data])
        force_magnitudes = [d["force_magnitude"] for d in self.force_data]
        force_in_direction = [d["force_in_direction"] for d in self.force_data]
        distances = [d["distance_from_start"] for d in self.force_data]
        ref_force = self.force_data[0]["reference_force"] if self.force_data else 0

        plt.figure(figsize=(15, 12))

        plt.subplot(2, 2, 1)
        plt.plot(timestamps, force_vectors[:, 0], label="Fx", linewidth=2, color="red")
        plt.plot(
            timestamps, force_vectors[:, 1], label="Fy", linewidth=2, color="green"
        )
        plt.plot(timestamps, force_vectors[:, 2], label="Fz", linewidth=2, color="blue")
        plt.title("Force Components vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Force magnitude and target
        plt.subplot(2, 2, 2)
        plt.plot(
            timestamps,
            force_magnitudes,
            label="Force Magnitude",
            linewidth=2,
            color="purple",
        )
        plt.plot(
            timestamps,
            force_in_direction,
            label="Force in Direction",
            linewidth=2,
            color="orange",
        )
        plt.axhline(
            y=ref_force,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Target ({ref_force}N)",
        )
        plt.title("Force Tracking")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Distance moved vs time
        plt.subplot(2, 2, 3)
        plt.plot(
            timestamps,
            distances,
            label="Distance from Start",
            linewidth=2,
            color="brown",
        )
        plt.axhline(
            y=self.distance_cap,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Distance Cap ({self.distance_cap}m)",
        )
        plt.title("Movement Distance vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Force error vs time
        plt.subplot(2, 2, 4)
        force_errors = [ref_force - abs(f) for f in force_in_direction]
        plt.plot(
            timestamps, force_errors, label="Force Error", linewidth=2, color="darkred"
        )
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        plt.axhline(
            y=self.min_force_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Threshold (±{self.min_force_threshold}N)",
        )
        plt.axhline(
            y=-self.min_force_threshold, color="orange", linestyle="--", linewidth=2
        )
        plt.title("Force Control Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Force Error (N)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/force_control_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Summary statistics
        final_force = force_in_direction[-1] if force_in_direction else 0
        max_distance = max(distances) if distances else 0
        control_duration = timestamps[-1] if timestamps else 0

        print(
            f"Force control plot saved as force_control_plot.png ({len(self.force_data)} samples)"
        )
        print(f"Control duration: {control_duration:.2f}s")
        print(f"Final force: {final_force:.2f}N (target: {ref_force}N)")
        print(f"Max distance moved: {max_distance:.3f}m (cap: {self.distance_cap}m)")
        print(f"Force error: {abs(ref_force - abs(final_force)):.2f}N")

    def disconnect(self):
        """Override disconnect to stop force control first"""
        self.stop_force_control()
        super().disconnect()


if __name__ == "__main__":
    robot = URForceController("192.168.1.66", hz=50, kp=0.015, ki=0.0, kd=0.000005)

    try:
        robot.go_home()

        robot.wait_for_commands()
        robot.wait_until_done()

        print("Current force reading:", robot.get_current_force())
        print("Force magnitude:", robot.get_force_magnitude())

        print("\nStarting force control...")
        robot.force_control_to_target(
            reference_force=10.0,
            direction=[0, 0, -1],
            distance_cap=0.2,
            timeout=30,
        )

        robot.wait_for_force_control()

        print("Force control complete!")
        print("Final force reading:", robot.get_current_force())

        robot.plot_force_data()

        print("Returning to home...")
        robot.go_home()
        robot.wait_for_commands()
        robot.wait_until_done()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot.disconnect()
        robot.plot()  # Plot robot motion data
        print("Robot disconnected")
