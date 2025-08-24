import numpy as np
from scipy.spatial.transform import Rotation as R


def _make_T(self, Rm, p):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T


def _Rt_p_from_T(self, T):
    return T[:3, :3], T[:3, 3]


def rvec_to_rotmat(rvec):
    return R.from_rotvec(rvec).as_matrix()


def rotmat_to_rvec(rotmat):
    return R.from_matrix(rotmat).as_rotvec()


def end_effector_rotation_from_normal(normal_vector):
    """
    Creates a 3x3 rotation matrix for an end-effector where:
    1. The end-effector's local Y-axis is aligned with the input normal_vector.
    2. The end-effector's local Z-axis points downward.
    3. Right-handed coordinate system: X = Y × Z
    """
    normal = np.array(normal_vector, dtype=float)
    normal = normal / np.linalg.norm(normal)  # Ensure unit vector

    local_y = normal

    global_z_up = np.array([0, 0, 1])

    if np.abs(np.dot(local_y, global_z_up)) > 0.99:
        reference = np.array([1, 0, 0])
    else:
        reference = global_z_up

    local_x = np.cross(reference, local_y)
    local_x = local_x / np.linalg.norm(local_x)

    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    if local_z[2] > 0:
        local_z = -local_z
        local_x = np.cross(local_y, local_z)
        local_x = local_x / np.linalg.norm(local_x)

    rotation_matrix = np.column_stack([local_x, local_y, local_z])

    det = np.linalg.det(rotation_matrix)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"Invalid rotation matrix generated! Determinant = {det}, expected ~1.0"
        )

    should_be_identity = rotation_matrix @ rotation_matrix.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
        raise ValueError("Invalid rotation matrix generated! Matrix is not orthogonal")

    return rotation_matrix


def wrap_angles(angle):
    """
    Wrap angles to [-π, π] range

    Args:
        angles: numpy array of angles in radians

    Returns:
        numpy array of wrapped angles in [-π, π] range
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
