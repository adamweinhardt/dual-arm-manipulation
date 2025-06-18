import zmq
import json
import numpy as np
from typing import List
from dataclasses import dataclass
import time
import os


@dataclass
class Box:
    id: int
    position: np.ndarray  # [x, y, z]
    rotation_matrix: np.ndarray  # 3x3
    confidence: float
    width: float
    height: float
    depth: float


def get_dimensions_from_config(board_config_path: str) -> tuple:
    """Get box dimensions from board config"""
    with open(board_config_path, "r") as f:
        board_data = json.load(f)

    min_pt = np.array([float("inf")] * 3)
    max_pt = np.array([float("-inf")] * 3)

    for marker in board_data["markers"]:
        for corner in marker["corners"]:
            for k in range(3):
                min_pt[k] = min(min_pt[k], corner[k])
                max_pt[k] = max(max_pt[k], corner[k])

    dims = max_pt - min_pt
    return dims[0], dims[1], dims[2]  # width, height, depth


class BoxTracker:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        self.port = config["port"]

        # Get config directory (go up one level from configs/)
        config_dir = os.path.dirname(os.path.dirname(config_path))

        # Get dimensions for each box
        self.box_dimensions = {}
        for i, board_config in enumerate(config["boards"]):
            # Resolve path relative to repo root
            if not os.path.isabs(board_config):
                board_config = os.path.join(config_dir, board_config.lstrip("./"))

            w, h, d = get_dimensions_from_config(board_config)
            self.box_dimensions[i] = (w, h, d)

        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Box tracker connected to port {self.port}")
        print(f"Tracking {len(self.box_dimensions)} box types")

    def get_boxes(self) -> List[Box]:
        """Get current box states - waits for message"""
        try:
            # Wait up to 100ms for a message
            self.socket.setsockopt(zmq.RCVTIMEO, 100)
            data = self.socket.recv_json()

            poses = data["poses"]
            boxes = []

            for box_id_str, pose_data in poses.items():
                box_id = int(box_id_str)
                if box_id in self.box_dimensions:
                    w, h, d = self.box_dimensions[box_id]

                    box = Box(
                        id=box_id,
                        position=np.array(pose_data["position"]),
                        rotation_matrix=np.array(pose_data["rotation_matrix"]),
                        confidence=pose_data["confidence"],
                        width=w,
                        height=h,
                        depth=d,
                    )
                    boxes.append(box)

            return boxes

        except zmq.Again:
            # Timeout - no message received
            return []

    def close(self):
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    config_path = "/home/weini/code/robot_ipc_control/configs/pose_estimation_config_single_camera_dual_arm.json"

    tracker = BoxTracker(config_path)

    try:
        while True:
            boxes = tracker.get_boxes()
            if boxes:
                print(f"Got {len(boxes)} boxes:")
                for box in boxes:
                    print(
                        f"  Box {box.id}: pos=[{box.position[0]:.3f}, {box.position[1]:.3f}, {box.position[2]:.3f}], dims=({box.width:.3f}, {box.height:.3f}, {box.depth:.3f}), conf={box.confidence:.2f}"
                    )
            else:
                print(".", end="", flush=True)

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tracker.close()
