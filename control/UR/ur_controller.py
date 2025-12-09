import time
import threading
import queue
from numpy import pi
import numpy as np
import zmq
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robot_ipc_control.pose_estimation.transform_utils import (
    rvec_to_rotmat,
    rotmat_to_rvec,
)

class URController(threading.Thread):
    def __init__(self, ip, hz=50):
        super().__init__(daemon=True)
        self.ip = ip

        if self.ip == "192.168.1.66":
            self.robot_id = 1
            self.robot_config = np.load(
                "/home/aweinhardt/Desktop/Thesis/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_right.npy"
            )
            self.port = 5559
            self.ee2marker_offset = np.array([0.00, 0.05753, -0.10, 0, 0, 0])
            self.ee2marker_offset_base = np.array([0.00, -0.05753, -0.10, 0, 0, 0])

        elif self.ip == "192.168.1.33":
            self.robot_id = 0
            self.robot_config = np.load(
                "/home/aweinhardt/Desktop/Thesis/dual-arm-manipulation/robot_ipc_control/calibration/base_pose_robot_left.npy"
            )
            self.port = 5556
            self.ee2marker_offset = np.array([0.00, -0.05753, -0.10, 0, 0, 0])
            self.ee2marker_offset_base = np.array([0.00, -0.05753, -0.10, 0, 0, 0])
        else:
            self.robot_id = None
            self.robot_config = None
            self.port = None
            self.ee2marker_offset = None

        self.rtde_control = RTDEControlInterface(self.ip)
        self.rtde_receive = RTDEReceiveInterface(self.ip)
        self.rtde_control.zeroFtSensor()
        
        # Command queue for threading
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()

        # State publisher
        self.hz = hz
        self.publishing = False
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://127.0.0.1:{self.port}")
        
        self.T_r2w = self.robot_config #4x4 hom transform
        self.R_r2w = np.array(self.T_r2w[:3, :3], dtype=float) #rotation 3x3
        self.T_w2r = np.linalg.inv(self.T_r2w)
        self.R_r2w_pin = np.array([[-1, 0, 0],
                                   [ 0,-1, 0],
                                   [ 0, 0, 1]])

        self.t_r2w = self.T_r2w[:3, 3] # +  np.array([0.00, -0.05753, -0.10]) #offset from tcp to my gripper. 3x1


        # Defaults
        # self.home_joints = [
        #     -pi / 2.0,
        #     -pi / 2.0,
        #     pi / 2.0,
        #     -pi / 2.0,
        #     -pi / 2.0,
        #     pi,
        # ]

        self.home_joints = [
            -1.7570222059832972,
            -2.4474531612791957,
            2.201507870350973,
            -1.374998615389206,
            -1.5552824179278772,
            2.9247472286224365,
        ]
        self.ee2marker = np.array(
            [-0.0064, 0.05753, -0.1149, -0.69923, -0.0101, -0.00407, 0.71481]
        )
        self.default_speed = 0.5
        self.default_acceleration = 0.25
        self.default_joint_speed = 1.0
        self.default_joint_acceleration = 0.5

        # Data recording
        self.previous_force = None
        self.previous_force_world = None
        self.alpha = 0.70
        self.data = []
        self.forces = []

        # Start threads
        self.start()

    def run(self):
        """Main control thread loop - processes movement commands"""

        while not self._stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)

                command()

                self.command_queue.task_done()
            except queue.Empty:
                continue

    def _start_publishing(self):
        """Start the state publishing thread"""

        self.publishing = True
        self.publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher_thread.start()

    def _record_during_movement(self):
        """Records robot state data while movement is happening"""
        time.sleep(0.1)

        for _ in range(100):
            state = self.get_state()
            self.data.append(state)
            time.sleep(0.05)

            if not self.is_moving():
                break

    def _publish_loop(self):
        """Main publishing loop - runs in background thread"""

        interval = 1.0 / self.hz

        def to_list(data):
            """Convert numpy arrays to lists for JSON serialization"""
            if hasattr(data, "tolist"):
                return data.tolist()
            elif isinstance(data, (list, tuple)):
                return list(data)
            else:
                return data

        while self.publishing:
            try:
                start_time = time.time()

                robot_state = self.get_state()

                if robot_state and self.publisher:
                    # convert np arrays to lists for JSON serialization
                    message = {
                        "timestamp": time.time(),
                        "robot_id": self.robot_id,
                        "Q": to_list(robot_state.get("joints", [])),
                        "pos": to_list(robot_state.get("pose", [])),
                        "vel": to_list(robot_state.get("speed", [])),
                        "force": to_list(robot_state.get("force", [])),
                        "filtered_force": to_list(
                            robot_state.get("filtered_force", [])
                        ),
                        "pose_world": to_list(robot_state.get("pose_world", [])),
                        "vel_world": to_list(robot_state.get("speed_world", [])),
                        "force_world": to_list(robot_state.get("force_world", [])),
                        "filtered_force_world": to_list(
                            robot_state.get("filtered_force_world", [])
                        ),
                        "gripper_world": to_list(robot_state.get("gripper_world", [])),
                    }

                    self.publisher.send_json(message)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Publishing error for {self.robot_id}: {e}")
                time.sleep(0.1)

    def world_2_robot(self, world_pose, robot_to_world):
        world_pose = np.array(world_pose)

        world_T = np.eye(4)
        world_T[:3, 3] = world_pose[:3]
        world_T[:3, :3] = rvec_to_rotmat(world_pose[3:])

        world_to_robot = np.linalg.inv(robot_to_world)
        robot_T = world_to_robot @ world_T

        robot_pos = robot_T[:3, 3]
        robot_rvec = rotmat_to_rvec(robot_T[:3, :3])

        return np.array(
            [
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_rvec[0],
                robot_rvec[1],
                robot_rvec[2],
            ]
        )
    
    def world_point_2_robot(self, world_point):
        """
        Transforms a 3D point [x, y, z] from World Frame to Robot Base Frame.
        """
        world_point_4d = np.append(world_point, 1.0)

        robot_to_world = self.T_r2w
        world_to_robot = np.linalg.inv(robot_to_world)

        local_point_4d = world_to_robot @ world_point_4d

        return local_point_4d[:3]
    
    def world_vector_2_robot(self, normal_world):
        """
        Transforms a 3D point [x, y, z] from World Frame to Robot Base Frame.
        """
        R_r2w = self.robot_config[:3, :3]   # rotation robot → world
        R_w2r = R_r2w.T                     # inverse rotation
        
        return R_w2r @ normal_world
    
    def robot_2_world(self, local_pose, robot_to_world):
        local_pose_6d = np.array(local_pose)

        local_T = np.eye(4)
        local_T[:3, 3] = local_pose_6d[:3]
        local_T[:3, :3] = rvec_to_rotmat(local_pose_6d[3:])

        world_T = robot_to_world @ local_T

        world_pos = world_T[:3, 3]
        world_rvec = rotmat_to_rvec(world_T[:3, :3])

        return np.array(
            [
                world_pos[0],
                world_pos[1],
                world_pos[2],
                world_rvec[0],
                world_rvec[1],
                world_rvec[2],
            ]
        )

    def moveL(self, pose_ee):
        command = lambda: self.rtde_control.moveL(
            pose_ee, self.default_speed, self.default_acceleration
        )
        self.command_queue.put(command)

    def moveL_world(self, pose_world):
        if isinstance(pose_world, np.ndarray):
            pose_world = pose_world.tolist()

        try:
            pose_ee = self.world_2_robot(pose_world, self.robot_config)

            self.moveL(pose_ee)

        except Exception as e:
            print(f"ERROR in moveL_world: {e}")

    def moveL_gripper_world(self, TCP_world):
        TCP_world = np.array(TCP_world)

        try:
            gripper_pose = TCP_world + self.ee2marker_offset

            self.moveL_world(gripper_pose)

        except Exception as e:
            print(f"ERROR in moveL_gripper_world: {e}")

    def speedL(self, speed_ee, acceleration=0.5, time_duration=0.1):
        command = lambda: self.rtde_control.speedL(
            speed_ee, acceleration, time_duration
        )
        self.command_queue.put(command)

    def speedL_world(self, speed_world, acceleration=0.5, time_duration=0.1):
        if isinstance(speed_world, np.ndarray):
            speed_world = speed_world.tolist()

        try:
            vel_world = np.array(speed_world[:3])
            omega_world = np.array(speed_world[3:6])

            world_to_robot = np.linalg.inv(self.robot_config)
            robot_rotation = world_to_robot[:3, :3]

            vel_robot_base = robot_rotation @ vel_world
            omega_robot_base = omega_world

            speed_command = [
                vel_robot_base[0],
                vel_robot_base[1],
                vel_robot_base[2],
                omega_robot_base[0],
                omega_robot_base[1],
                omega_robot_base[2],
            ]

            self.speedL(
                speed_command, acceleration=acceleration, time_duration=time_duration
            )

        except Exception as e:
            print(f"ERROR in speedL_world: {e}")

    def get_J_pin(self, J):
        """Rotate Jacobian from robot-base to world frame."""
        J_world = np.zeros_like(J)
        J_world[:3, :] = self.R_r2w_pin @ J[:3, :]
        J_world[3:, :] = self.R_r2w_pin @ J[3:, :]
        return J_world

    def speedStop(self):
        command = lambda: self.rtde_control.speedStop()
        self.command_queue.put(command)

    def moveJ(self, joints):
        command = lambda: self.rtde_control.moveJ(
            joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self.command_queue.put(command)

    def speedJ(self, joint_speed, dt):
        command = lambda: self.rtde_control.speedJ(
            joint_speed, self.default_joint_acceleration, dt*4
        )
        self.command_queue.put(command)

    def go_home(self):
        command = lambda: self.rtde_control.moveJ(
            self.home_joints, self.default_joint_speed, self.default_joint_acceleration
        )
        self.command_queue.put(command)

    def force_low_pass_filter(self, previous_force, current_force, alpha):
        if previous_force is None:
            return current_force
        return (1 - alpha) * current_force + alpha * previous_force

    def get_state(self):
        """Get robot state with detailed timing breakdown"""

        pose_robot_base = self.rtde_receive.getActualTCPPose()

        force_robot_base = np.array(self.rtde_receive.getActualTCPForce())

        speed_robot_base = np.array(self.rtde_receive.getActualTCPSpeed())

        joints = self.rtde_receive.getActualQ()

        # === COORDINATE TRANSFORMATIONS ===
        pose_world = self.robot_2_world(pose_robot_base, self.robot_config)

        rotation_robot_to_world = self.robot_config[:3, :3]
        linear_speed_world = rotation_robot_to_world @ speed_robot_base[:3]
        angular_speed_world = rotation_robot_to_world @ speed_robot_base[3:]
        speed_world = np.concatenate((linear_speed_world, angular_speed_world))

        force_vector_world = rotation_robot_to_world @ force_robot_base[:3]
        torque_vector_world = rotation_robot_to_world @ force_robot_base[3:]
        force_world = np.concatenate((force_vector_world, torque_vector_world))

        # === FILTERING ===
        filtered_force = self.force_low_pass_filter(
            self.previous_force, force_robot_base, self.alpha
        )
        self.previous_force = filtered_force

        filtered_force_world = self.force_low_pass_filter(
            self.previous_force_world, force_vector_world, self.alpha
        )
        self.previous_force_world = filtered_force_world

        gripper_base = pose_robot_base - self.ee2marker_offset_base
        gripper_world = pose_world - self.ee2marker_offset

        self.forces.append([force_vector_world, filtered_force_world])

        return {
            "pose": pose_robot_base,
            "pose_world": pose_world,
            "joints": joints,
            "speed": speed_robot_base,
            "speed_world": speed_world,
            "force": force_robot_base,
            "force_world": force_world,
            "torque_world": torque_vector_world,
            "filtered_force": filtered_force,
            "filtered_force_world": filtered_force_world,
            "gripper_world": gripper_world,
            "gripper_base": gripper_base,
        }

    def is_moving(self):
        speeds = self.rtde_receive.getActualTCPSpeed()
        return np.linalg.norm(speeds, 2) > 0.01

    def wait_for_commands(self):
        self.command_queue.join()

    def wait_until_done(self):
        time.sleep(0.2)

        if not self.is_moving():
            state = self.get_state()
            self.data.append(state)

    def disconnect(self):
        self._stop_event.set()
        self.join(timeout=2)
        self.rtde_control.disconnect()
        self.rtde_receive.disconnect()



