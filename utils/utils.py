# utils/transforms.py

import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# -------------------------------------------------
# Basic SE(3) helpers
# -------------------------------------------------


def make_T(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform T from rotation Rm (3x3) and translation t (3,)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(Rm, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def decompose_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) from homogeneous transform T (4x4)."""
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform T (4x4)."""
    Rm, t = decompose_T(T)
    Tinv = np.eye(4)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


# -------------------------------------------------
# Rotation conversions
# -------------------------------------------------


def rvec_to_rotmat(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vector (3,) -> rotation matrix (3x3)."""
    return R.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()


def rotmat_to_rvec(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> Rodrigues rotation vector (3,)."""
    return R.from_matrix(np.asarray(rotmat, dtype=float)).as_rotvec()


# -------------------------------------------------
# Pose <-> Transform
# -------------------------------------------------


def pose6_to_T(pose6: np.ndarray) -> np.ndarray:
    """Pose [x,y,z, rx,ry,rz] -> homogeneous transform (4x4)."""
    pose6 = np.asarray(pose6, dtype=float)
    T = np.eye(4)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = rvec_to_rotmat(pose6[3:])
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    """Homogeneous transform (4x4) -> Pose [x,y,z, rx,ry,rz]."""
    Rm, t = decompose_T(T)
    rvec = rotmat_to_rvec(Rm)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2]], dtype=float)


# -------------------------------------------------
# Transformations of different quantities
# -------------------------------------------------


def transform_point(T_a2b: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    """Transform a point from frame a to frame b using T_a2b."""
    Rm, t = decompose_T(T_a2b)
    return Rm @ np.asarray(p_a, dtype=float) + t


def transform_rotation(T_a2b: np.ndarray, R_a: np.ndarray | np.ndarray) -> np.ndarray:
    """
    Transform a rotation from frame a to b.
    Input can be rotation matrix (3x3) or Rodrigues rvec (3,).
    Output type matches input type.
    """
    Rm, _ = decompose_T(T_a2b)
    if np.shape(R_a) == (3,):
        R_b = Rm @ rvec_to_rotmat(R_a)
        return rotmat_to_rvec(R_b)
    else:
        return Rm @ R_a


def transform_pose(T_a2b: np.ndarray, pose_a: np.ndarray) -> np.ndarray:
    """Transform a 6D pose [x,y,z, rx,ry,rz] from frame a to b."""
    T_pose_a = pose6_to_T(pose_a)
    T_pose_b = T_a2b @ T_pose_a
    return T_to_pose6(T_pose_b)


def transform_twist(T_a2b: np.ndarray, twist_a: np.ndarray) -> np.ndarray:
    """
    Transform a twist [vx,vy,vz, wx,wy,wz] from frame a to b.
    Assumes same origin, only rotates v and ω.
    """
    Rm, _ = decompose_T(T_a2b)
    twist_a = np.asarray(twist_a, dtype=float)
    v_b = Rm @ twist_a[:3]
    w_b = Rm @ twist_a[3:]
    return np.concatenate([v_b, w_b])


def transform_wrench(T_a2b: np.ndarray, wrench_a: np.ndarray) -> np.ndarray:
    """
    Transform a wrench [Fx,Fy,Fz, Mx,My,Mz] from frame a to b.
    Assumes same origin, only rotates F and M.
    """
    Rm, _ = decompose_T(T_a2b)
    wrench_a = np.asarray(wrench_a, dtype=float)
    f_b = Rm @ wrench_a[:3]
    m_b = Rm @ wrench_a[3:]
    return np.concatenate([f_b, m_b])


# -------------------------------------------------
# Utility
# -------------------------------------------------


def wrap_angles(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π]."""
    angle = np.asarray(angle, dtype=float)
    return np.arctan2(np.sin(angle), np.cos(angle))


def _assert_rotmat(name, M):
    A = np.asarray(M)
    if A.shape != (3, 3):
        print(f"[ASSERT ROTMAT FAIL] {name} shape={A.shape} -> converting if possible")
        return _as_rotmat(A)
    return A


def _as_rotmat(maybe_rot):
    """Accept a 3x3 rotation matrix or a 3-vector rotvec; return 3x3 matrix."""
    arr = np.asarray(maybe_rot)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (3,):
        return R.from_rotvec(arr).as_matrix()
    raise ValueError(f"Expected rotmat (3,3) or rotvec (3,), got {arr.shape}")


def end_effector_rotation_from_normal(normal_vector, eps=1e-9):
    """
    TCP rotation with columns [x, y, z] such that:
      - y points into the surface (aligned with normal_vector, normalized)
      - z is the closest vector to global -Z while orthogonal to y
      - x = y × z (right-handed)
    """
    n = np.asarray(normal_vector, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < eps:
        raise ValueError("normal_vector has near-zero magnitude")
    y = n / n_norm  # y -> into surface

    g_down = np.array([0.0, 0.0, -1.0])  # global down

    # Project global down onto plane orthogonal to y (best-possible 'down' given y)
    z = g_down - np.dot(g_down, y) * y
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        # y is (anti)parallel to global down; pick a horizontal fallback
        # choose x-axis as fallback, then re-project
        fallback = np.array([1.0, 0.0, 0.0])
        z = fallback - np.dot(fallback, y) * y
        z_norm = np.linalg.norm(z)
        if z_norm < eps:
            # extremely degenerate: pick y-orthogonal basis directly
            # choose any vector not collinear with y
            fallback = np.array([0.0, 1.0, 0.0])
            z = fallback - np.dot(fallback, y) * y
            z_norm = np.linalg.norm(z)
    z = z / z_norm

    x = np.cross(y, z)  # right-handed (z = x × y => x = y × z)
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        raise ValueError("Failed to construct orthonormal basis")
    x = x / x_norm

    # Re-orthogonalize z to kill any numerical drift and ensure z = x × y
    z = np.cross(x, y)

    R = np.column_stack([x, y, z])

    # sanity checks
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError("Rotation not orthogonal")
    if np.linalg.det(R) < 0.999999:
        raise ValueError("Rotation not right-handed (det != 1)")
    return R


def _canonicalize_pair(pA, pB):
    d = pB - pA
    axis = int(np.argmax(np.abs(d)))  # 0=x, 1=y, 2=z (dominant separation)
    if d[axis] < 0:  # enforce positive along dominant axis
        pA, pB = pB, pA
    return pA, pB

def _compute_initial_box_frame(pA, pB):
    # --- canonicalize ordering so both arms build the SAME frame ---
    pA, pB = _canonicalize_pair(pA, pB)

    x_hat = (pB - pA) / (np.linalg.norm(pB - pA) + 1e-9)

    # project world-z onto plane orthogonal to x_hat; fallback if degenerate
    z_guess = np.array([0, 0, 1])
    z_hat = z_guess - np.dot(z_guess, x_hat) * x_hat
    if np.linalg.norm(z_hat) < 1e-6:
        z_guess = np.array([0, 1, 0])
        z_hat = z_guess - np.dot(z_guess, x_hat) * x_hat
    z_hat /= np.linalg.norm(z_hat) + 1e-9

    y_hat = np.cross(z_hat, x_hat)
    y_hat /= np.linalg.norm(y_hat) + 1e-9  # (tiny numeric guard)

    R_WB0 = np.column_stack([x_hat, y_hat, z_hat])
    p_WB0 = 0.5 * (pA + pB)
    return R_WB0, p_WB0

def _freeze_sparsity(A, eps=1e-12):
    # Replace exact zeros with a tiny epsilon to keep nnz pattern constant.
    A = np.asarray(A, dtype=float).copy()
    A[A == 0.0] = eps
    return A

def _as_rowvec_1d(x, name, length=None):
    a = np.asarray(x, dtype=float).squeeze()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D after squeeze, got {a.shape}")
    if length is not None and a.shape[0] != length:
        raise ValueError(f"{name} length {a.shape[0]} != {length}")
    return a

def _as_2d(x, name, shape0=None, shape1=None):
    a = np.asarray(x, dtype=float).squeeze()
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeeze, got {a.shape}")
    if shape0 is not None and a.shape[0] != shape0:
        raise ValueError(f"{name} shape[0] {a.shape[0]} != {shape0}")
    if shape1 is not None and a.shape[1] != shape1:
        raise ValueError(f"{name} shape[1] {a.shape[1]} != {shape1}")
    return a

def diag6(vals):
    a = np.asarray(vals, dtype=float).reshape(-1)
    if a.size == 1:
        return np.eye(6) * a.item()
    if a.size == 3:
        return np.diag([a[0], a[1], a[2], a[0], a[1], a[2]])
    if a.size == 6:
        return np.diag(a)
    raise ValueError("diag6 expects 1, 3, or 6 values")

def is_finite(*arrays):
    return all(np.all(np.isfinite(np.asarray(x))) for x in arrays)

def short_arc_log(R_ref, R_cur):
    q_ref = R.from_matrix(R_ref).as_quat()
    q_cur = R.from_matrix(R_cur).as_quat()
    if np.dot(q_ref, q_cur) < 0.0:
        q_ref = -q_ref
    R_rel = (R.from_quat(q_ref) * R.from_quat(q_cur).inv()).as_matrix()
    return pin.log3(R_rel)