#!/usr/bin/env python3
# tests/compare_yours_vs_pin.py
import numpy as np
import pinocchio as pin

from control.impedance_qp_controller import URImpedanceController  # <-- fix if needed

URDF_PATH = "ur5/UR5e.urdf"
FRAME_NAME = "tool0"

# ---------- tiny helpers (no external args needed by you) ----------
def _rel_err(A, B):
    na = np.linalg.norm(A)
    return np.linalg.norm(A - B) / (1.0 if na == 0 else na)

def _is_sym(A, tol=1e-9):
    return np.linalg.norm(A - A.T) <= tol

def _numeric_jacobian(model, data, frame_id, q):
    nv = model.nv
    J = np.zeros((6, nv))
    zero = np.zeros(nv)
    for k in range(nv):
        qdot = zero.copy(); qdot[k] = 1.0
        pin.forwardKinematics(model, data, q, qdot, zero)
        pin.updateFramePlacements(model, data)
        v = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # linear first, angular second
        J[:, k] = np.r_[v.linear, v.angular]
    # reset FK
    pin.forwardKinematics(model, data, q, zero, zero)
    pin.updateFramePlacements(model, data)
    return J

def _pin_Lambda_svd_damped(J, M, lam2=1e-6):
    Minv = np.linalg.inv(M)
    A = J @ Minv @ J.T
    A = 0.5 * (A + A.T)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_dinv = S / (S**2 + lam2)   # Tikhonov in inverse space
    Lam = (Vt.T * S_dinv) @ U.T
    Lam = 0.5 * (Lam + Lam.T)
    cond = (S.max() / S.min()) if S.min() > 0 else np.inf
    return Lam, S.min(), S.max(), cond

def _projector_residuals(A, Lam):
    R1 = A @ Lam @ A - A
    R2 = Lam @ A @ Lam - Lam
    r1 = np.linalg.norm(R1) / max(1.0, np.linalg.norm(A))
    r2 = np.linalg.norm(R2) / max(1.0, np.linalg.norm(Lam))
    return r1, r2

# ---------- main test (no args) ----------
def main():
    print(f"Loading model: {URDF_PATH}")
    model = pin.buildModelFromUrdf(URDF_PATH)
    data  = model.createData()
    frame_id = model.getFrameId(FRAME_NAME)
    print(f"Using frame '{FRAME_NAME}' (id={frame_id})")

    # --- create YOUR object without calling __init__ (no robot/net deps) ---
    yours = URImpedanceController.__new__(URImpedanceController)
    # seed only what your methods need
    yours.pin_model = model
    yours.pin_data = model.createData()
    yours.pin_frame_id = frame_id

    # a few test configurations
    qs = [
        np.array([0.0, -np.pi/2,  np.pi/2, 0.0, 0.0, 0.0]),
        np.array([0.2, -1.0,      1.2,     -0.3, 0.5, -0.2]),
        np.array([-0.8, -1.3,     1.6,      0.9, -0.4, 0.7]),
    ]

    for idx, q in enumerate(qs, 1):
        print(f"\n=== CONFIG {idx} q={np.array2string(q, precision=3, floatmode='fixed')} ===")
        # --- J: Pin vs Numeric vs Yours ---
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J_pin = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_num = _numeric_jacobian(model, data, frame_id, q)
        J_you = yours.get_J(q)

        e_pin_num = _rel_err(J_pin, J_num)
        e_you_pin = _rel_err(J_you, J_pin)
        e_you_num = _rel_err(J_you, J_num)
        print(f"[J]  rel(pin,num)={e_pin_num:.3e}  rel(yours,pin)={e_you_pin:.3e}  rel(yours,num)={e_you_num:.3e}")

        # --- M: Pin vs Yours ---
        M_pin = pin.crba(model, data, q); M_pin = 0.5*(M_pin + M_pin.T)
        M_you = yours.get_M(q)
        eM = _rel_err(M_you, M_pin)
        eigs = np.linalg.eigvalsh(M_pin)
        print(f"[M]  rel(yours,pin)={eM:.3e}  eig(min,max)=({eigs.min():.3e},{eigs.max():.3e})  sym? pin={_is_sym(M_pin)} yours={_is_sym(M_you)}")

        # --- Λ: Pin (SVD-damped) vs Yours (your actual implementation) ---
        Lam_pin, smin, smax, cond = _pin_Lambda_svd_damped(J_pin, M_pin, lam2=1e-6)
        # use YOUR get_Lambda exactly as you wrote it
        Lam_you = yours.get_Lambda(J_you, M_you)   # your method does A += eps*I then inv

        # report & sanity
        eLam = _rel_err(Lam_you, Lam_pin)
        print(f"[Λ]  cond(A)~{cond:.3e}  rel(yours, pin_damped)={eLam:.3e}  sym? pin={_is_sym(Lam_pin)} yours={_is_sym(Lam_you)}")
        # residuals (use the same A built from Pin for a fair projector test)
        Minv = np.linalg.inv(M_pin)
        A = J_pin @ Minv @ J_pin.T
        r1, r2 = _projector_residuals(0.5*(A+A.T), Lam_you)
        print(f"[Λ]  projector residuals (with YOUR Λ):  ||AΛA−A||/||A||={r1:.3e},  ||ΛAΛ−Λ||/||Λ||={r2:.3e}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
