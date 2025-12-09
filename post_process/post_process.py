import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from scipy.spatial.transform import Rotation

class PostProcess:
    """
    Minimal post-process viewer for .npz files from save_everything().
    - No helper methods.
    - __init__ only keeps the path and loaded npz reference.
    - plot_PID() and plot_data3D() pull keys on the fly and degrade gracefully.
    - No files saved, just plt.show().
    """

    def __init__(self, npz_path):
        self.path = npz_path
        self.npz  = np.load(npz_path, allow_pickle=True)

    def evaluate_PID(self, left_npz_path: str, right_npz_path: str):
        print("\n================ PID TRACKING EVALUATION ================")

        def _evaluate_one(path: str, label: str):
            data = np.load(path, allow_pickle=True)
            print(f"\n=========== PID ARM {label} ===========")

            # ----- FORCE -----
            if "force_vector" in data and "reference_force_vector" in data:
                A = np.asarray(data["force_vector"], dtype=float)
                B = np.asarray(data["reference_force_vector"], dtype=float)
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Force] samples      : {n}")
                    print(f"RMSE Fx,Fy,Fz        : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [N]")
                    print(f"RMSE ||F - F_ref||   : {rmse_norm:.4f} N")
            else:
                print("\n[Force] series not found (measured or reference).")

            # ----- POSITION -----
            if "position" in data and "reference_position" in data:
                A = np.asarray(data["position"], dtype=float)
                B = np.asarray(data["reference_position"], dtype=float)
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Position] samples   : {n}")
                    print(f"RMSE x,y,z           : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m]")
                    print(f"RMSE ||p - p_ref||   : {rmse_norm:.4f} m")
            else:
                print("\n[Position] series not found (measured or reference).")

            # ----- ROTATION (correct: via relative rotation log-map) -----
            if "rotation" in data and "reference_rotation" in data:
                r = np.asarray(data["rotation"], dtype=float)                # rotvec (N,3)
                r_ref = np.asarray(data["reference_rotation"], dtype=float)  # rotvec (N,3)
                n = min(len(r), len(r_ref))
                if n > 0:
                    r = r[:n]; r_ref = r_ref[:n]
                    # relative rotation error: R_err = R_ref^T * R_meas
                    Rm = Rotation.from_rotvec(r)
                    Rr = Rotation.from_rotvec(r_ref)
                    e_r = (Rr.inv() * Rm).as_rotvec()     # log-map error (N,3)
                    # RMSE on components and on angle (norm)
                    rmse_axes = np.sqrt(np.mean(e_r**2, axis=0))
                    ang = np.linalg.norm(e_r, axis=1)
                    rmse_norm = float(np.sqrt(np.mean(ang**2)))  # == angle RMSE
                    print(f"\n[Rotation] samples   : {n}")
                    print(f"RMSE rotvec x,y,z    : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad]")
                    print(f"RMSE ||r - r_ref||   : {rmse_norm:.4f} rad")
                    print(f"RMSE rotation angle  : {rmse_norm:.4f} rad")
            else:
                print("\n[Rotation] series not found (measured or reference).")

            # ----- LINEAR VELOCITY -----
            if "linear_velocity" in data and "reference_linear_velocity" in data:
                A = np.asarray(data["linear_velocity"], dtype=float)
                B = np.asarray(data["reference_linear_velocity"], dtype=float)
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Linear Velocity] samples : {n}")
                    print(f"RMSE vx,vy,vz        : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m/s]")
                    print(f"RMSE ||v - v_ref||   : {rmse_norm:.4f} m/s")
            else:
                print("\n[Linear Velocity] series not found (measured or reference).")

            # ----- ANGULAR VELOCITY -----
            if "angular_velocity" in data and "reference_angular_velocity" in data:
                A = np.asarray(data["angular_velocity"], dtype=float)
                B = np.asarray(data["reference_angular_velocity"], dtype=float)
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Angular Velocity] samples : {n}")
                    print(f"RMSE wx,wy,wz        : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad/s]")
                    print(f"RMSE ||w - w_ref||   : {rmse_norm:.4f} rad/s")
            else:
                print("\n[Angular Velocity] series not found (measured or reference).")

            # controller frequency info
            Hz = data["meta__Hz"] if "meta__Hz" in data else (data["meta__control_rate_hz"] if "meta__control_rate_hz" in data else None)
            if Hz is not None:
                print(f"\n[Meta] Logged controller Hz: {float(Hz):.2f}")

        _evaluate_one(left_npz_path,  "LEFT")
        _evaluate_one(right_npz_path, "RIGHT")
        print("\n=========================================================\n")


    def evaluate_QP(self, qp_path: str):
        """
        Evaluate dual-arm QP tracking (force, position, rotation, velocity, angular velocity)
        from one .npz produced by DualArmImpedanceAdmittanceQP.save_everything().

        Rotation RMSE is computed from the relative rotation log-map:
            e_r = log(R_ref^T R_meas), with
            component RMSE on e_r and angle RMSE = RMSE(||e_r||).
        """
        print("\n================ QP TRACKING EVALUATION =================")

        try:
            data = np.load(qp_path, allow_pickle=True)
        except Exception as e:
            print(f"Could not load '{qp_path}': {e}")
            return

        print(f"file: {qp_path}")

        for arm in ("L", "R"):
            print(f"\n=========== QP ARM {arm} ===========")

            # --- Force ---
            kF, kFref = f"F_{arm}_vec", f"F_{arm}_ref"
            if kF in data and kFref in data:
                A = np.asarray(data[kF], dtype=float)
                B = np.asarray(data[kFref], dtype=float)
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Force] samples      : {n}")
                    print(f"RMSE Fx,Fy,Fz        : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [N]")
                    print(f"RMSE ||F - F_ref||   : {rmse_norm:.4f} N")
            else:
                print("\n[Force] missing force logs for this arm.")

            # --- Position ---
            kp, kpref = f"p_{arm}", f"p_ref_{arm}"
            if kp in data and kpref in data:
                A = np.asarray(data[kp], dtype=float)[..., :3]
                B = np.asarray(data[kpref], dtype=float)[..., :3]
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Position] samples   : {n}")
                    print(f"RMSE x,y,z           : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m]")
                    print(f"RMSE ||p - p_ref||   : {rmse_norm:.4f} m")
            else:
                print("\n[Position] missing p / p_ref for this arm.")

            # --- Rotation (correct: via relative rotation log-map) ---
            kr, krref = f"rvec_{arm}", f"rvec_ref_{arm}"
            if kr in data and krref in data:
                r = np.asarray(data[kr], dtype=float)
                r_ref = np.asarray(data[krref], dtype=float)
                n = min(len(r), len(r_ref))
                if n > 0:
                    r = r[:n]; r_ref = r_ref[:n]
                    Rm = Rotation.from_rotvec(r)
                    Rr = Rotation.from_rotvec(r_ref)
                    e_r = (Rr.inv() * Rm).as_rotvec()   # (n,3)
                    rmse_axes = np.sqrt(np.mean(e_r**2, axis=0))
                    ang = np.linalg.norm(e_r, axis=1)
                    rmse_angle = float(np.sqrt(np.mean(ang**2)))  # == norm RMSE
                    print(f"\n[Rotation] samples   : {n}")
                    print(f"RMSE rotvec x,y,z    : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad]")
                    print(f"RMSE ||r - r_ref||   : {rmse_angle:.4f} rad")
                    print(f"RMSE rotation angle  : {rmse_angle:.4f} rad")
            else:
                print("\n[Rotation] missing rvec / rvec_ref for this arm.")

            # --- Linear velocity ---
            kv, kvref = f"v_{arm}", f"v_ref_{arm}"
            if kv in data and kvref in data:
                A = np.asarray(data[kv], dtype=float)[..., :3]
                B = np.asarray(data[kvref], dtype=float)[..., :3]
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Linear Velocity] samples : {n}")
                    print(f"RMSE vx,vy,vz         : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m/s]")
                    print(f"RMSE ||v - v_ref||    : {rmse_norm:.4f} m/s")
            else:
                print("\n[Linear Velocity] missing v / v_ref for this arm.")

            # --- Angular velocity ---
            kw, kwref = f"w_{arm}", f"w_ref_{arm}"
            if kw in data and kwref in data:
                A = np.asarray(data[kw], dtype=float)[..., :3]
                B = np.asarray(data[kwref], dtype=float)[..., :3]
                n = min(len(A), len(B))
                if n > 0:
                    A = A[:n]; B = B[:n]
                    if A.ndim == 1: A = A[:, None]
                    if B.ndim == 1: B = B[:, None]
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Angular Velocity] samples: {n}")
                    print(f"RMSE wx,wy,wz         : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad/s]")
                    print(f"RMSE ||w - w_ref||    : {rmse_norm:.4f} rad/s")
            else:
                print("\n[Angular Velocity] missing w / w_ref for this arm.")

        print("\n=========================================================\n")

    def evaluate_QP2(self, qp_path: str, skip_seconds: float = 0.0,
                    sample_rate_hz: float | None = None,
                    time_key: str | None = None):
        """
        Evaluate dual-arm QP tracking (force, position, rotation, velocity, angular velocity)
        from one .npz produced by DualArmImpedanceAdmittanceQP.save_everything().

        Parameters
        ----------
        qp_path : str
            Path to the saved .npz log.
        skip_seconds : float, optional
            Ignore the first `skip_seconds` seconds of the trajectory when computing RMSE.
            If a time vector is present (see `time_key` or auto-detected), it will be used.
            Otherwise uses `dt` in the file or `sample_rate_hz` if provided. Default 0.0.
        sample_rate_hz : float, optional
            Fallback sampling rate (Hz) if no time vector or dt is found.
        time_key : str, optional
            Name of the time vector in the .npz (e.g., "t"). If None, tries common names:
            ["t", "time", "times", "timestamp", "timestamps", "qp_time", "log_time"].

        Notes
        -----
        Rotation RMSE is computed from the relative rotation log-map:
            e_r = log(R_ref^T R_meas), with
            component RMSE on e_r and angle RMSE = RMSE(||e_r||).
        """
        print("\n================ QP TRACKING EVALUATION =================")

        try:
            data = np.load(qp_path, allow_pickle=True)
        except Exception as e:
            print(f"Could not load '{qp_path}': {e}")
            return

        print(f"file: {qp_path}")

        # --- determine trim index from skip_seconds ---
        def _compute_i0():
            if skip_seconds <= 0:
                return 0, None

            # 1) explicit time_key if provided
            if time_key is not None and time_key in data:
                t = np.asarray(data[time_key]).reshape(-1)
                t0 = float(t[0])
                target = t0 + float(skip_seconds)
                i0 = int(np.searchsorted(t, target, side="left"))
                return max(i0, 0), ("time", time_key, t0)

            # 2) try common time keys
            for k in ["t", "time", "times", "timestamp", "timestamps", "qp_time", "log_time"]:
                if k in data:
                    t = np.asarray(data[k]).reshape(-1)
                    t0 = float(t[0])
                    target = t0 + float(skip_seconds)
                    i0 = int(np.searchsorted(t, target, side="left"))
                    return max(i0, 0), ("time", k, t0)

            # 3) use dt if present
            for k in ["dt", "DT", "sample_dt"]:
                if k in data:
                    try:
                        dt = float(np.asarray(data[k]).reshape(()))
                        if dt > 0:
                            i0 = int(np.floor(skip_seconds / dt + 1e-12))
                            return max(i0, 0), ("dt", k, dt)
                    except Exception:
                        pass

            # 4) fallback to provided sample rate
            if sample_rate_hz is not None and sample_rate_hz > 0:
                i0 = int(np.floor(skip_seconds * float(sample_rate_hz)))
                return max(i0, 0), ("Hz", "sample_rate_hz", float(sample_rate_hz))

            # 5) nothing found
            return 0, None

        i0, trim_basis = _compute_i0()
        if skip_seconds > 0:
            if trim_basis is None:
                print(f"[Trim] Requested skip_seconds={skip_seconds:.3f}s, "
                    f"but no time/dt/Hz found. Proceeding without trimming.")
                i0 = 0
            else:
                kind, key, val = trim_basis
                if kind == "time":
                    print(f"[Trim] Skipping first {skip_seconds:.3f}s using time vector '{key}' "
                        f"(t0={val:.6f}).")
                elif kind == "dt":
                    print(f"[Trim] Skipping first {skip_seconds:.3f}s using dt '{key}'={val:.6g}s "
                        f"(i0={i0}).")
                elif kind == "Hz":
                    print(f"[Trim] Skipping first {skip_seconds:.3f}s using sample_rate_hz={val:.6g} "
                        f"(i0={i0}).")

        # helper: align, trim, and return (A,B,n)
        def _prep(A, B):
            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)
            n = min(len(A), len(B))
            if n <= 0:
                return None
            # apply trim
            if i0 >= n:
                # everything trimmed away
                return None
            A = A[:n]; B = B[:n]
            A = A[i0:]; B = B[i0:]
            if A.ndim == 1: A = A[:, None]
            if B.ndim == 1: B = B[:, None]
            return A, B, len(A)

        for arm in ("L", "R"):
            print(f"\n=========== QP ARM {arm} ===========")

            # --- Force ---
            kF, kFref = f"F_{arm}_vec", f"F_{arm}_ref"
            if kF in data and kFref in data:
                prepped = _prep(data[kF], data[kFref])
                if prepped is not None:
                    A, B, n = prepped
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Force] samples      : {n}")
                    print(f"RMSE Fx,Fy,Fz        : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [N]")
                    print(f"RMSE ||F - F_ref||   : {rmse_norm:.4f} N")
                else:
                    print("\n[Force] no samples remain after trimming.")
            else:
                print("\n[Force] missing force logs for this arm.")

            # --- Position ---
            kp, kpref = f"p_{arm}", f"p_ref_{arm}"
            if kp in data and kpref in data:
                prepped = _prep(np.asarray(data[kp])[..., :3], np.asarray(data[kpref])[..., :3])
                if prepped is not None:
                    A, B, n = prepped
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Position] samples   : {n}")
                    print(f"RMSE x,y,z           : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m]")
                    print(f"RMSE ||p - p_ref||   : {rmse_norm:.4f} m")
                else:
                    print("\n[Position] no samples remain after trimming.")
            else:
                print("\n[Position] missing p / p_ref for this arm.")

            # --- Rotation (via relative rotation log-map) ---
            kr, krref = f"rvec_{arm}", f"rvec_ref_{arm}"
            if kr in data and krref in data:
                prepped = _prep(data[kr], data[krref])
                if prepped is not None:
                    r, r_ref, n = prepped
                    Rm = Rotation.from_rotvec(r)
                    Rr = Rotation.from_rotvec(r_ref)
                    e_r = (Rr.inv() * Rm).as_rotvec()   # (n,3)
                    rmse_axes = np.sqrt(np.mean(e_r**2, axis=0))
                    ang = np.linalg.norm(e_r, axis=1)
                    rmse_angle = float(np.sqrt(np.mean(ang**2)))  # == norm RMSE
                    print(f"\n[Rotation] samples   : {n}")
                    print(f"RMSE rotvec x,y,z    : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad]")
                    print(f"RMSE ||r - r_ref||   : {rmse_angle:.4f} rad")
                    print(f"RMSE rotation angle  : {rmse_angle:.4f} rad")
                else:
                    print("\n[Rotation] no samples remain after trimming.")
            else:
                print("\n[Rotation] missing rvec / rvec_ref for this arm.")

            # --- Linear velocity ---
            kv, kvref = f"v_{arm}", f"v_ref_{arm}"
            if kv in data and kvref in data:
                prepped = _prep(np.asarray(data[kv])[..., :3], np.asarray(data[kvref])[..., :3])
                if prepped is not None:
                    A, B, n = prepped
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Linear Velocity] samples : {n}")
                    print(f"RMSE vx,vy,vz         : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [m/s]")
                    print(f"RMSE ||v - v_ref||    : {rmse_norm:.4f} m/s")
                else:
                    print("\n[Linear Velocity] no samples remain after trimming.")
            else:
                print("\n[Linear Velocity] missing v / v_ref for this arm.")

            # --- Angular velocity ---
            kw, kwref = f"w_{arm}", f"w_ref_{arm}"
            if kw in data and kwref in data:
                prepped = _prep(np.asarray(data[kw])[..., :3], np.asarray(data[kwref])[..., :3])
                if prepped is not None:
                    A, B, n = prepped
                    err = A - B
                    rmse_axes = np.sqrt(np.mean(err**2, axis=0))
                    rmse_norm = float(np.sqrt(np.mean(np.linalg.norm(err, axis=1)**2)))
                    print(f"\n[Angular Velocity] samples: {n}")
                    print(f"RMSE wx,wy,wz         : [{rmse_axes[0]:.4f}, {rmse_axes[1]:.4f}, {rmse_axes[2]:.4f}]  [rad/s]")
                    print(f"RMSE ||w - w_ref||    : {rmse_norm:.4f} rad/s")
                else:
                    print("\n[Angular Velocity] no samples remain after trimming.")
            else:
                print("\n[Angular Velocity] missing w / w_ref for this arm.")

        print("\n=========================================================\n")

    def plot_PID(self):
        # pull what we need, skip if missing
        T = np.asarray(self.npz["timestamp"]) if "timestamp" in self.npz else np.array([])
        if T.size == 0:
            print("No PID data (timestamp missing).")
            return

        def get2d(k):
            if k not in self.npz: 
                return np.zeros((0,3))
            a = np.asarray(self.npz[k])
            if a.ndim == 1: a = a[:, None]
            return a

        force_p = get2d("force_p_term"); force_i = get2d("force_i_term"); force_d = get2d("force_d_term")
        pose_p  = get2d("pos_p_term");   pose_i  = get2d("pos_i_term");   pose_d  = get2d("pos_d_term")
        ff_p    = get2d("ff_pos_p_term");ff_i    = get2d("ff_pos_i_term");ff_d    = get2d("ff_pos_d_term")

        if force_p.shape[0] == 0 and pose_p.shape[0] == 0 and ff_p.shape[0] == 0:
            print("No PID term arrays found in file.")
            return

        # sum (match shortest length/cols)
        def safe_sum(a,b,c):
            shapes = [x.shape for x in (a,b,c) if x.size>0]
            if not shapes: return np.zeros((0,3))
            N = min(s[0] for s in shapes)
            C = min(s[1] for s in shapes)
            out = np.zeros((N,C))
            if a.size: out += a[:N,:C]
            if b.size: out += b[:N,:C]
            if c.size: out += c[:N,:C]
            return out

        sum_p = safe_sum(force_p, pose_p, ff_p)
        sum_i = safe_sum(force_i, pose_i, ff_i)
        sum_d = safe_sum(force_d, pose_d, ff_d)

        # gain strings (optional)
        def fmt(prefix):
            kp = self.npz.get(f"meta__{prefix}_pid_kp", None)
            ki = self.npz.get(f"meta__{prefix}_pid_ki", None)
            kd = self.npz.get(f"meta__{prefix}_pid_kd", None)
            if kp is None or ki is None or kd is None: return f"{prefix.upper()} PID: (n/a)"
            kp,ki,kd = np.array(kp).flatten(), np.array(ki).flatten(), np.array(kd).flatten()
            if kp.size==1: return f"{prefix.upper()} PID: (Kp={kp[0]:.3f}, Ki={ki[0]:.3f}, Kd={kd[0]:.3f})"
            return f"{prefix.upper()} PID: (Kp={kp}, Ki={ki}, Kd={kd})"
        title_force = fmt("force"); title_pose = fmt("pose"); title_ff = fmt("ff_pose")

        colors = ["red","green","blue"]; labels = ["X","Y","Z"]
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        plt.figure(figsize=(28,14))
        plt.suptitle(
            "PID Components (Force / Pose / Feed-Forward) and Sums\n"
            f"{title_force} | {title_pose} | {title_ff}\nViewed: {now}",
            fontsize=16, y=0.98
        )

        def block(idx, data, title):
            plt.subplot(3,4,idx)
            if data.size:
                m = min(3, data.shape[1])
                n = min(len(T), data.shape[0])
                for i in range(m):
                    plt.plot(T[:n], data[:n, i], label=labels[i], linewidth=2, color=colors[i])
                plt.legend()
            plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Output"); plt.grid(True, alpha=0.3)
            plt.axhline(0, color="black", lw=1, alpha=0.5)

        block(1, force_p, "Force PID — P"); block(2, pose_p, "Pose PID — P"); block(3, ff_p, "FF Pose PID — P"); block(4, sum_p, "SUM P")
        block(5, force_i, "Force PID — I"); block(6, pose_i, "Pose PID — I"); block(7, ff_i, "FF Pose PID — I"); block(8, sum_i, "SUM I")
        block(9, force_d, "Force PID — D"); block(10,pose_d, "Pose PID — D"); block(11,ff_d, "FF Pose PID — D"); block(12,sum_d, "SUM D")

        plt.tight_layout(rect=[0,0.03,1,0.93])
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        plt.tight_layout(rect=[0,0.03,1,0.93])
        pid_file = f"plots/pid_terms_postprocess_{current_datetime}.png"
        plt.savefig(pid_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {pid_file}")

    def plot_data3D(self):
        T = np.asarray(self.npz["timestamp"]) if "timestamp" in self.npz else np.array([])
        if T.size == 0:
            print("No control timeline in file.")
            return

        def get2d(k):
            if k not in self.npz: return np.zeros((0,3))
            a = np.asarray(self.npz[k])
            if a.ndim == 1: a = a[:, None]
            return a

        # --- vectors from npz ---
        forces      = get2d("force_vector")
        pos         = get2d("position")
        ref_pos     = get2d("reference_position")
        rotations   = get2d("rotation")
        ref_rots    = get2d("reference_rotation")
        ref_forces  = get2d("reference_force_vector")

        f_err = get2d("force_error_vector")
        p_err = get2d("position_error_vector")
        r_err = get2d("rotation_error_vector")

        f_out = get2d("force_output_vector")
        p_out = get2d("position_output_vector")
        ff_po = get2d("ff_position_output_vector")
        tot_o = get2d("total_output_vector")
        r_out = get2d("rotation_output_vector")

        # --- make ref_pos 3D if it's 6D ---
        if ref_pos.shape[-1] == 6: ref_pos = ref_pos[:, :3]
        elif ref_pos.size and ref_pos.shape[-1] != 3: ref_pos = ref_pos[:, :3]

        # --- helper norms ---
        def row_norm(a):
            a = np.asarray(a)
            if a.size == 0: return np.zeros(T.shape)
            if a.ndim == 1: a = a[:, None]
            return np.sqrt((a*a).sum(axis=1))

        force_mag     = row_norm(forces)
        ref_force_mag = row_norm(ref_forces)

        # --- start pos ---
        if "meta__start_position" in self.npz:
            start_pos = np.asarray(self.npz["meta__start_position"]).reshape(-1)
        elif pos.size:
            start_pos = pos[0]
        else:
            start_pos = np.zeros(3)
        distances = row_norm(pos - start_pos)

        # --- control direction (fallback z) ---
        ctrl_dir = np.asarray(self.npz["control_direction"]).reshape(-1) if "control_direction" in self.npz else np.array([0,0,1.0])
        ctrl_dir = ctrl_dir / (np.linalg.norm(ctrl_dir) + 1e-9)

        pos_err_dir   = (p_err @ ctrl_dir) if p_err.size else np.zeros_like(T)
        force_err_dir = (f_err @ ctrl_dir) if f_err.size else np.zeros_like(T)

        f_err_mag  = row_norm(f_err);  p_err_mag  = row_norm(p_err)
        f_out_mag  = row_norm(f_out);  p_out_mag  = row_norm(p_out)
        ff_po_mag  = row_norm(ff_po);  tot_o_mag  = row_norm(tot_o)
        r_err_mag  = row_norm(r_err);  r_out_mag  = row_norm(r_out)

        has_rot = rotations.size and ref_rots.size

        # --- deadzone (from npz; accept scalar or array) ---
        dz_val = None
        if "deadzone_threshold" in self.npz:
            dz_raw = np.asarray(self.npz["deadzone_threshold"])
            try:
                dz_val = float(dz_raw[0] if dz_raw.ndim > 0 else dz_raw.item() if hasattr(dz_raw, "item") else dz_raw)
            except Exception:
                dz_val = None
            if (dz_val is not None) and (not np.isfinite(dz_val) or dz_val <= 0):
                dz_val = None  # treat non-positive/NaN as "do not plot"

        # --- titles ---
        import datetime, os
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ref_force_str = f"{ref_force_mag[0]:.2f}N" if ref_force_mag.size else "N/A"
        initial_target_pose_str = (
            f"[{ref_pos[0,0]:.3f}, {ref_pos[0,1]:.3f}, {ref_pos[0,2]:.3f}]" if ref_pos.size else "[N/A]"
        )

        def gains_str(base):
            kp = self.npz.get(f"meta__{base}_pid_kp", None)
            ki = self.npz.get(f"meta__{base}_pid_ki", None)
            kd = self.npz.get(f"meta__{base}_pid_kd", None)
            if kp is None or ki is None or kd is None: return "n/a"
            kp,ki,kd = np.array(kp).flatten(), np.array(ki).flatten(), np.array(kd).flatten()
            return f"Kp={kp[0]:.3f}, Ki={ki[0]:.3f}, Kd={kd[0]:.3f}" if kp.size==1 else f"Kp={kp}, Ki={ki}, Kd={kd}"
        force_pid_str = gains_str("force")
        pose_pid_str  = gains_str("pose")

        import matplotlib.pyplot as plt
        fig_width = 24 if has_rot else 18
        plt.figure(figsize=(fig_width, 12))
        n_cols = 4 if has_rot else 3

        plt.suptitle("\n".join([
            f"{'6DOF' if has_rot else '3DOF'} Vector Control Data (Viewed {now})",
            f"Ref Force: {ref_force_str}, Target Pose: {initial_target_pose_str}",
            f"Force PID: ({force_pid_str}), Pose PID: ({pose_pid_str})",
        ]), fontsize=14, y=0.98)

        colors = ["red","green","blue"]

        # ===== Row 1 =====
        # Force vs target
        plt.subplot(3, n_cols, 1)
        if forces.size:
            for a,c in enumerate(colors[:min(3, forces.shape[1])]):
                plt.plot(T[:forces.shape[0]], forces[:forces.shape[0], a], label=f"{'XYZ'[a]} Force", lw=2, color=c)
        if ref_forces.size:
            for a,c in enumerate(colors[:min(3, ref_forces.shape[1])]):
                plt.plot(T[:ref_forces.shape[0]], ref_forces[:ref_forces.shape[0], a], "--", label=f"{'XYZ'[a]} Target", lw=2, color=c, alpha=0.7)
        plt.title("Force vs Target Force"); plt.xlabel("Time (s)"); plt.ylabel("Force (N)"); plt.legend(); plt.grid(True, alpha=0.3)

        # Position vs target
        plt.subplot(3, n_cols, 2)
        if pos.size:
            for a,c in enumerate(colors[:min(3, pos.shape[1])]):
                plt.plot(T[:pos.shape[0]], pos[:pos.shape[0], a], label=f"{'XYZ'[a]} Position", lw=2, color=c)
        if ref_pos.size:
            for a,c in enumerate(colors[:min(3, ref_pos.shape[1])]):
                plt.plot(T[:ref_pos.shape[0]], ref_pos[:ref_pos.shape[0], a], "--", label=f"{'XYZ'[a]} Target", lw=2, color=c, alpha=0.7)
        plt.title("Position vs Target Position"); plt.xlabel("Time (s)"); plt.ylabel("Position (m)"); plt.legend(); plt.grid(True, alpha=0.3)

        if has_rot:
            plt.subplot(3, n_cols, 3)
            if rotations.size:
                for a,c in enumerate(colors[:min(3, rotations.shape[1])]):
                    plt.plot(T[:rotations.shape[0]], rotations[:rotations.shape[0], a], label=f"R{'xyz'[a]} Current", lw=2, color=c)
            if ref_rots.size:
                for a,c in enumerate(colors[:min(3, ref_rots.shape[1])]):
                    plt.plot(T[:ref_rots.shape[0]], ref_rots[:ref_rots.shape[0], a], "--", label=f"R{'xyz'[a]} Target", lw=2, color=c, alpha=0.7)
            plt.title("Rotation vs Target Rotation"); plt.xlabel("Time (s)"); plt.ylabel("Rotation (rad)"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(3, n_cols, 4)
        else:
            plt.subplot(3, n_cols, 3)

        plt.plot(T[:distances.shape[0]], distances, label="Distance from Start", lw=2, color="brown")
        if "meta__distance_cap" in self.npz:
            cap = float(self.npz["meta__distance_cap"])
            if not np.isnan(cap):
                plt.axhline(y=cap, color="red", ls="--", lw=2, label=f"Distance Cap ({cap}m)")
        plt.title("Movement Distance vs Time"); plt.xlabel("Time (s)"); plt.ylabel("Distance (m)"); plt.legend(); plt.grid(True, alpha=0.3)

        # ===== Row 2 =====
        # Force error
        plt.subplot(3, n_cols, n_cols + 1)
        if f_err.size:
            for a,c in enumerate(colors[:min(3, f_err.shape[1])]):
                plt.plot(T[:f_err.shape[0]], f_err[:f_err.shape[0], a], label=f"Force Err {'XYZ'[a]}", lw=2, color=c)
            plt.plot(T[:f_err_mag.shape[0]], f_err_mag, "--", label="|Force Err|", lw=2, color="black")
        plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Force Error Vectors"); plt.xlabel("Time (s)"); plt.ylabel("Force Error (N)"); plt.legend(); plt.grid(True, alpha=0.3)

        # Position error (+ optional deadzone lines)
        plt.subplot(3, n_cols, n_cols + 2)
        if p_err.size:
            for a,c in enumerate(colors[:min(3, p_err.shape[1])]):
                plt.plot(T[:p_err.shape[0]], p_err[:p_err.shape[0], a], label=f"Pos Err {'XYZ'[a]}", lw=2, color=c)
            plt.plot(T[:p_err_mag.shape[0]], p_err_mag, "--", label="|Pos Err|", lw=2, color="black")
        plt.axhline(0, color="black", lw=1, alpha=0.5)
        if dz_val is not None:
            plt.axhline(y= dz_val, color="orange", linestyle="dotted", linewidth=2, label=f"Deadzone +{dz_val:.3f} m")
            plt.axhline(y=-dz_val, color="orange", linestyle="dotted", linewidth=2, label=f"Deadzone -{dz_val:.3f} m")
        plt.title("3D Position Error Vectors"); plt.xlabel("Time (s)"); plt.ylabel("Position Error (m)"); plt.legend(); plt.grid(True, alpha=0.3)

        if has_rot:
            plt.subplot(3, n_cols, n_cols + 3)
            if r_err.size:
                for a,c in enumerate(colors[:min(3, r_err.shape[1])]):
                    plt.plot(T[:r_err.shape[0]], r_err[:r_err.shape[0], a], label=f"Rot Err R{'xyz'[a]}", lw=2, color=c)
                plt.plot(T[:r_err_mag.shape[0]], r_err_mag, "--", label="|Rot Err|", lw=2, color="black")
            plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Rotation Error Vectors"); plt.xlabel("Time (s)"); plt.ylabel("Rotation Error (rad)"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(3, n_cols, n_cols + 4)
        else:
            plt.subplot(3, n_cols, n_cols + 3)

        plt.plot(T[:force_err_dir.shape[0]], force_err_dir, label="Force Error in Direction", lw=2, color="darkred")
        plt.plot(T[:pos_err_dir.shape[0]],   pos_err_dir,   label="Position Error in Direction", lw=2, color="darkgreen")
        plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("Errors in Control Direction"); plt.xlabel("Time (s)"); plt.ylabel("Error"); plt.legend(); plt.grid(True, alpha=0.3)

        # ===== Row 3 =====
        plt.subplot(3, n_cols, 2*n_cols + 1)
        if f_out.size:
            for a,c in enumerate(colors[:min(3, f_out.shape[1])]):
                plt.plot(T[:f_out.shape[0]], f_out[:f_out.shape[0], a], label=f"Force Out {'XYZ'[a]}", lw=2, color=c)
            plt.plot(T[:f_out_mag.shape[0]], f_out_mag, "--", label="|Force Out|", lw=2, color="black")
        plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Force Control Outputs"); plt.xlabel("Time (s)"); plt.ylabel("Force Output"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(3, n_cols, 2*n_cols + 2)
        if p_out.size:
            for a,c in enumerate(colors[:min(3, p_out.shape[1])]):
                plt.plot(T[:p_out.shape[0]], p_out[:p_out.shape[0], a], label=f"PID Out {'XYZ'[a]}", lw=2, color=c)
        if ff_po.size:
            m = min(3, ff_po.shape[1])
            labs = ["FF Out X","FF Out Y","FF Out Z"]
            cols = ["magenta","orange","cyan"]
            for a in range(m):
                plt.plot(T[:ff_po.shape[0]], ff_po[:ff_po.shape[0], a], label=labs[a], lw=2, color=cols[a])
            plt.plot(T[:p_out_mag.shape[0]],  p_out_mag,  label="|PID Out|", lw=2, color="black")
            plt.plot(T[:ff_po_mag.shape[0]], ff_po_mag, label="|FF Out|",  lw=2, color="purple")
        plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Position Control Outputs (PID + FF)"); plt.xlabel("Time (s)"); plt.ylabel("Position Output"); plt.legend(ncol=2, fontsize=9); plt.grid(True, alpha=0.3)

        if has_rot:
            plt.subplot(3, n_cols, 2*n_cols + 3)
            if r_out.size:
                for a,c in enumerate(colors[:min(3, r_out.shape[1])]):
                    plt.plot(T[:r_out.shape[0]], r_out[:r_out.shape[0], a], label=f"Rot Out R{'xyz'[a]}", lw=2, color=c)
                plt.plot(T[:r_out_mag.shape[0]], r_out_mag, "--", label="|Rot Out|", lw=2, color="black")
            plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Rotation Control Outputs"); plt.xlabel("Time (s)"); plt.ylabel("Rotation Output (rad/s)"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(3, n_cols, 2*n_cols + 4)
        else:
            plt.subplot(3, n_cols, 2*n_cols + 3)

        if tot_o.size:
            for a,c in enumerate(colors[:min(3, tot_o.shape[1])]):
                plt.plot(T[:tot_o.shape[0]], tot_o[:tot_o.shape[0], a], label=f"Total Out {'XYZ'[a]}", lw=2, color=c)
            plt.plot(T[:tot_o_mag.shape[0]], tot_o_mag, "--", label="|Total Out|", lw=2, color="purple")
        plt.axhline(0, color="black", lw=1, alpha=0.5); plt.title("3D Total Linear Outputs"); plt.xlabel("Time (s)"); plt.ylabel("Total Output"); plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0,0.03,1,0.95])
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        plot_type = "6dof" if has_rot else "3dof"
        file_path = f"plots/comprehensive_{plot_type}_control_postprocess_{current_datetime}.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {file_path}")

    def plot_pid_tracking(self, title_prefix="PID_Taskspace"):
        import os, datetime
        import numpy as np
        import matplotlib.pyplot as plt

        data = self.npz

        # time
        if "timestamp" not in data:
            print("[plot_pid_tracking] No 'timestamp' in npz.")
            return
        t = np.asarray(data["timestamp"]).reshape(-1)

        def _get(name):
            return np.asarray(data[name]) if name in data else None

        p      = _get("position")
        p_ref  = _get("reference_position")
        r_v    = _get("rotation")                 # rotvec
        r_vref = _get("reference_rotation")

        v      = _get("linear_velocity")
        v_ref  = _get("reference_linear_velocity")
        w      = _get("angular_velocity")
        w_ref  = _get("reference_angular_velocity")

        colors = ["r", "g", "b"]
        comp   = ["X", "Y", "Z"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

        # --- 1) Position vs ref ---
        ax = axes[0, 0]
        if p is not None:
            n = min(len(t), len(p))
            for i_c in range(min(3, p.shape[1])):
                ax.plot(t[:n], p[:n, i_c], color=colors[i_c], label=f"p{comp[i_c]}")
        if p_ref is not None:
            n = min(len(t), len(p_ref))
            for i_c in range(min(3, p_ref.shape[1])):
                ax.plot(t[:n], p_ref[:n, i_c], "--", color=colors[i_c], alpha=0.7, label=f"p{comp[i_c]} ref")
        ax.set_ylabel("Position [m]")
        ax.set_title("Position vs Reference")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

        # --- 2) Rotation (rotvec) vs ref ---
        ax = axes[0, 1]
        if r_v is not None:
            n = min(len(t), len(r_v))
            for i_c in range(min(3, r_v.shape[1])):
                ax.plot(t[:n], r_v[:n, i_c], color=colors[i_c], label=f"r{comp[i_c]}")
        if r_vref is not None:
            n = min(len(t), len(r_vref))
            for i_c in range(min(3, r_vref.shape[1])):
                ax.plot(t[:n], r_vref[:n, i_c], "--", color=colors[i_c], alpha=0.7, label=f"r{comp[i_c]} ref")
        ax.set_ylabel("Rot. Vector [rad]")
        ax.set_title("Rotation (rotvec) vs Reference")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

        # --- 3) Position error norm ---
        ax = axes[0, 2]
        if p is not None and p_ref is not None:
            n = min(len(p), len(p_ref), len(t))
            e_p = p[:n] - p_ref[:n]
            err_norm = np.linalg.norm(e_p, axis=1)
            ax.plot(t[:n], err_norm, label="‖p - p_ref‖")
        ax.set_ylabel("Error [m]")
        ax.set_title("Position Error Norm")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # --- 4) Linear velocity vs ref ---
        ax = axes[1, 0]
        if v is not None:
            n = min(len(t), len(v))
            for i_c in range(min(3, v.shape[1])):
                ax.plot(t[:n], v[:n, i_c], color=colors[i_c], label=f"v{comp[i_c]}")
        if v_ref is not None:
            n = min(len(t), len(v_ref))
            for i_c in range(min(3, v_ref.shape[1])):
                ax.plot(t[:n], v_ref[:n, i_c], "--", color=colors[i_c], alpha=0.7, label=f"v{comp[i_c]} ref")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Lin. Vel. [m/s]")
        ax.set_title("Linear Velocity vs Reference")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

        # --- 5) Angular velocity vs ref ---
        ax = axes[1, 1]
        if w is not None:
            n = min(len(t), len(w))
            for i_c in range(min(3, w.shape[1])):
                ax.plot(t[:n], w[:n, i_c], color=colors[i_c], label=f"ω{comp[i_c]}")
        if w_ref is not None:
            n = min(len(t), len(w_ref))
            for i_c in range(min(3, w_ref.shape[1])):
                ax.plot(t[:n], w_ref[:n, i_c], "--", color=colors[i_c], alpha=0.7, label=f"ω{comp[i_c]} ref")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Ang. Vel. [rad/s]")
        ax.set_title("Angular Velocity vs Reference")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

        # --- 6) Rotation error angle ---
        ax = axes[1, 2]
        if r_v is not None and r_vref is not None:
            n = min(len(r_v), len(r_vref), len(t))
            Rm     = Rotation.from_rotvec(r_v[:n])
            Rm_ref = Rotation.from_rotvec(r_vref[:n])
            R_rel  = Rm_ref.inv() * Rm
            ang_err = R_rel.magnitude()
            ax.plot(t[:n], ang_err, label="rotation angle error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [rad]")
        ax.set_title("Rotation Error Angle")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.suptitle(f"{title_prefix} (viewed {now})", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs("plots", exist_ok=True)
        fname = f"plots/{title_prefix.lower()}_{now}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot_pid_tracking] Saved {fname}")

    def plot_box_path_PID(self):
        # --- load measured box pose ---
        if "box_positions" not in self.npz or "box_rotmats" not in self.npz:
            print("Missing measured box pose: need 'box_positions' and 'box_rotmats'.")
            return
        pos_meas = np.asarray(self.npz["box_positions"])
        R_meas   = np.asarray(self.npz["box_rotmats"])

        # --- load reference box pose ---
        has_ref = False
        if "box_ref_positions" in self.npz and "box_ref_rotmats" in self.npz:
            pos_ref = np.asarray(self.npz["box_ref_positions"])
            R_ref   = np.asarray(self.npz["box_ref_rotmats"])
            has_ref = True
        elif ("traj_rotation_matrices" in self.npz and
            "frame_R_WB0" in self.npz and "frame_p_WB0" in self.npz):
            R_WB0 = np.asarray(self.npz["frame_R_WB0"]).reshape(3,3)
            p_WB0 = np.asarray(self.npz["frame_p_WB0"]).reshape(3,)
            rot_updates = np.asarray(self.npz["traj_rotation_matrices"])
            pos_off = np.asarray(self.npz["traj_position_offsets"]) \
                if "traj_position_offsets" in self.npz else None
            K = len(rot_updates)
            pos_ref = np.zeros((K,3))
            R_ref   = np.zeros((K,3,3))
            for i in range(K):
                R_B0B = rot_updates[i]
                R_WB  = R_WB0 @ R_B0B
                R_ref[i] = R_WB
                dp_B0 = pos_off[i] if pos_off is not None and i < len(pos_off) else np.zeros(3)
                pos_ref[i] = p_WB0 + (R_WB0 @ dp_B0)
            has_ref = True

        # --- time base ---
        T = np.asarray(self.npz.get("timestamp", np.arange(len(pos_meas))))
        if T.size == 0: T = np.arange(len(pos_meas))
        if has_ref:
            T_ref = np.linspace(T[0], T[-1], num=len(pos_ref))
        else:
            T_ref = None

        # --- rotmat → roll/pitch/yaw ---
        def rot_to_rpy(Rstack):
            try:
                return Rotation.from_matrix(Rstack).as_euler('xyz', degrees=True)
            except Exception:
                eul = np.zeros((len(Rstack),3))
                r11, r21, r31 = Rstack[:,0,0], Rstack[:,1,0], Rstack[:,2,0]
                r32, r33 = Rstack[:,2,1], Rstack[:,2,2]
                eul[:,0] = np.degrees(np.arctan2(r32, r33))   # roll
                eul[:,1] = np.degrees(np.arcsin(-r31))        # pitch
                eul[:,2] = np.degrees(np.arctan2(r21, r11))   # yaw
                return eul

        rpy_meas = rot_to_rpy(R_meas)
        rpy_ref  = rot_to_rpy(R_ref) if has_ref else None

        # --- plot ---
        os.makedirs("plots", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/box_path_meas_vs_ref_{stamp}.png"

        plt.figure(figsize=(14, 8))
        colors = ["red","green","blue"]
        labels = ["X","Y","Z"]

        # Position vs time
        plt.subplot(2,1,1)
        for i,c in enumerate(colors):
            plt.plot(T[:len(pos_meas)], pos_meas[:len(T), i], label=f"meas {labels[i]}", lw=2, color=c)
            if has_ref:
                plt.plot(T_ref, pos_ref[:, i], "--", lw=2, color=c, alpha=0.7, label=f"ref {labels[i]}")
        plt.title("Box Position vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Orientation (RPY) vs time
        plt.subplot(2,1,2)
        rpy_labels = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        for i,c in enumerate(colors):
            plt.plot(T[:len(rpy_meas)], rpy_meas[:len(T), i], label=f"meas {rpy_labels[i]}", lw=2, color=c)
            if has_ref:
                plt.plot(T_ref, rpy_ref[:, i], "--", lw=2, color=c, alpha=0.7, label=f"ref {rpy_labels[i]}")
        plt.title("Box Orientation (RPY) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {fname}")

    def plot_taskspace(self, arm: str, title_prefix="TaskspaceTracking"):
        arm = arm.upper()
        ts = np.asarray(self.npz.get("t_epoch", []))
        if ts.size == 0:
            print("[plot_taskspace] No time axis found (t_epoch).")
            return
        t = ts - ts[0]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        colors = plt.cm.tab10(np.arange(3))
        labels = ["X","Y","Z"]

        # pull arrays (measured, reference, errors)
        P        = np.asarray(self.npz.get(f"p_{arm}",        np.zeros((0,3))))
        V        = np.asarray(self.npz.get(f"v_{arm}",        np.zeros((0,3))))
        RVEC     = np.asarray(self.npz.get(f"rvec_{arm}",     np.zeros((0,3))))
        W        = np.asarray(self.npz.get(f"w_{arm}",        np.zeros((0,3))))
        P_ref    = np.asarray(self.npz.get(f"p_ref_{arm}",    np.zeros((0,3))))
        V_ref    = np.asarray(self.npz.get(f"v_ref_{arm}",    np.zeros((0,3))))
        RVEC_ref = np.asarray(self.npz.get(f"rvec_ref_{arm}", np.zeros((0,3))))
        W_ref    = np.asarray(self.npz.get(f"w_ref_{arm}",    np.zeros((0,3))))
        E_p      = np.asarray(self.npz.get(f"e_p_{arm}",      np.zeros((0,3))))
        E_v      = np.asarray(self.npz.get(f"e_v_{arm}",      np.zeros((0,3))))
        E_w      = np.asarray(self.npz.get(f"e_w_{arm}",      np.zeros((0,3))))
        E_r      = np.asarray(self.npz.get(f"e_r_{arm}",      np.zeros((0,3))))

        if P.shape[0] == 0:
            print(f"[plot_taskspace] No TCP data for arm {arm}.")
            return

        E_p_mag = np.linalg.norm(E_p, axis=1) if E_p.size else np.zeros_like(t)
        E_v_mag = np.linalg.norm(E_v, axis=1) if E_v.size else np.zeros_like(t)
        E_w_mag = np.linalg.norm(E_w, axis=1) if E_w.size else np.zeros_like(t)
        E_r_mag = np.linalg.norm(E_r, axis=1) if E_r.size else np.zeros_like(t)

        fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
        fig.suptitle(f"{title_prefix} Robot_{arm} – {stamp}", fontsize=14, y=0.99)

        ax = axes[0,0]
        for k in range(3):
            if P.shape[0]:     ax.plot(t[:P.shape[0]],     P[:,k],     color=colors[k], label=f"p{labels[k]}")
            if P_ref.shape[0]: ax.plot(t[:P_ref.shape[0]], P_ref[:,k], "--", color=colors[k], alpha=0.9, label=f"p{labels[k]} ref")
        ax.set_title("Position vs Ref"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[0,1]
        for k in range(3):
            if E_p.shape[0]: ax.plot(t[:E_p.shape[0]], E_p[:,k], color=colors[k], label=f"e_p{labels[k]}")
        ax.plot(t[:E_p_mag.shape[0]], E_p_mag, "-", lw=2, color="black", label="‖e_p‖")
        ax.set_title("Translational Error"); ax.set_ylabel("m"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[1,0]
        for k in range(3):
            if RVEC.shape[0]:     ax.plot(t[:RVEC.shape[0]],     RVEC[:,k],     color=colors[k], label=f"r{labels[k]}")
            if RVEC_ref.shape[0]: ax.plot(t[:RVEC_ref.shape[0]], RVEC_ref[:,k], "--", color=colors[k], alpha=0.9, label=f"r{labels[k]} ref")
        ax.set_title("Rotation (rotvec) vs Ref"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[1,1]
        for k in range(3):
            if E_r.shape[0]: ax.plot(t[:E_r.shape[0]], E_r[:,k], color=colors[k], label=f"e_r{labels[k]}")
        ax.plot(t[:E_r_mag.shape[0]], E_r_mag, "-", lw=2, color="black", label="‖e_r‖")
        ax.set_title("Rotational Error"); ax.set_ylabel("rad"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[2,0]
        for k in range(3):
            if V.shape[0]:     ax.plot(t[:V.shape[0]],     V[:,k],     color=colors[k], label=f"v{labels[k]}")
            if V_ref.shape[0]: ax.plot(t[:V_ref.shape[0]], V_ref[:,k], "--", color=colors[k], alpha=0.9, label=f"v{labels[k]} ref")
        ax.set_title("Linear Velocity vs Ref"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=3, fontsize=8)

        ax = axes[2,1]
        for k in range(3):
            if E_v.shape[0]: ax.plot(t[:E_v.shape[0]], E_v[:,k], color=colors[k], label=f"e_v{labels[k]}")
        ax.plot(t[:E_v_mag.shape[0]], E_v_mag, "-", lw=2, color="black", label="‖e_v‖")
        ax.set_title("Linear-Velocity Error"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        ax = axes[3, 0]
        for k in range(3):
            if W.shape[0]:
                ax.plot(t[:W.shape[0]], W[:, k], color=colors[k], label=f"w{labels[k]}")
            if W_ref.shape[0]:
                ax.plot(t[:W_ref.shape[0]], W_ref[:, k], "--", color=colors[k], alpha=0.9, label=f"w{labels[k]} ref")

        ax.set_title("Angular Velocity vs Ref")
        ax.set_xlabel("s")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)


        ax = axes[3,1]
        for k in range(3):
            if E_w.shape[0]: ax.plot(t[:E_w.shape[0]], E_w[:,k], color=colors[k], label=f"e_w{labels[k]}")
        ax.plot(t[:E_w_mag.shape[0]], E_w_mag, "-", lw=2, color="black", label="‖e_w‖")
        ax.set_title("Angular-Velocity Error"); ax.set_xlabel("s"); ax.set_ylabel("rad/s"); ax.grid(True, alpha=0.3); ax.legend(ncol=4, fontsize=8)

        plt.tight_layout(rect=[0, 0.035, 1, 0.97])
        fname = f"plots/{title_prefix.lower()}_{arm}_{stamp}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[SAVED] {fname}")

    # ========================== JOINTSPACE (L/R) ==========================
    def plot_jointspace(self, arm: str, title_prefix="Joint_Space"):
        arm = arm.upper()
        ts = np.asarray(self.npz.get("t_epoch", []))
        if ts.size == 0:
            print("[plot_jointspace] No time axis found.")
            return
        t = ts - ts[0]
        q     = np.asarray(self.npz.get(f"q_{arm}",      np.zeros((0,6))))
        qdot  = np.asarray(self.npz.get(f"q_dot_{arm}",  np.zeros((0,6))))
        qddot = np.asarray(self.npz.get(f"q_ddot_{arm}", np.zeros((0,6))))
        if q.shape[0] == 0:
            print(f"[plot_jointspace] No joint data for arm {arm}.")
            return

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        n = q.shape[1]
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"{title_prefix} Robot_{arm} – {stamp}", fontsize=13, y=0.98)
        for j in range(n):
            axs[0].plot(t[:q.shape[0]], q[:, j], label=f"q{j+1}")
            axs[1].plot(t[:qdot.shape[0]], qdot[:, j], label=f"q̇{j+1}")
            axs[2].plot(t[:qddot.shape[0]], qddot[:, j], label=f"q̈{j+1}")
        axs[0].set_ylabel("Position [rad]");  axs[0].legend(fontsize=7, ncol=3); axs[0].grid(True, alpha=0.3)
        axs[1].set_ylabel("Velocity [rad/s]");axs[1].legend(fontsize=7, ncol=3); axs[1].grid(True, alpha=0.3)
        axs[2].set_ylabel("Acceleration [rad/s²]"); axs[2].legend(fontsize=7, ncol=3); axs[2].grid(True, alpha=0.3)
        axs[2].set_xlabel("Time [s]")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"plots/{title_prefix.lower()}_{arm}_{stamp}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[SAVED] {fname}")

    # ====================== QP OBJECTIVE BREAKDOWN =======================
    def plot_qp_objective(self, title_prefix="QP_Objective_Breakdown"):
        ts = np.asarray(self.npz.get("t_epoch", []))
        if ts.size == 0:
            print("[plot_qp_objective] No time axis.")
            return
        t = ts - ts[0]
        imp_L = np.asarray(self.npz.get("obj_imp_L", np.zeros(0)))
        imp_R = np.asarray(self.npz.get("obj_imp_R", np.zeros(0)))
        adm_L = np.asarray(self.npz.get("obj_grasp_L", np.zeros(0)))
        adm_R = np.asarray(self.npz.get("obj_grasp_R", np.zeros(0)))
        reg   = np.asarray(self.npz.get("obj_reg", np.zeros(0)))
        tot   = np.asarray(self.npz.get("obj_total", np.zeros(0)))
        if tot.size == 0:
            print("[plot_qp_objective] No objective breakdown arrays.")
            return

        fig, ax = plt.subplots(1,1, figsize=(12,6))
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)
        ax.plot(t[:tot.size], tot, label="Total", linewidth=2.5, color="black", alpha=0.8)
        if imp_L.size: ax.plot(t[:imp_L.size], imp_L, label="Imp L", linestyle="-", linewidth=1.5, alpha=0.8)
        if imp_R.size: ax.plot(t[:imp_R.size], imp_R, label="Imp R", linestyle="-", linewidth=1.5, alpha=0.8)
        if adm_L.size: ax.plot(t[:adm_L.size], adm_L, label="Adm L", linestyle="--", linewidth=1.5, alpha=0.8)
        if adm_R.size: ax.plot(t[:adm_R.size], adm_R, label="Adm R", linestyle="--", linewidth=1.5, alpha=0.8)
        if reg.size:   ax.plot(t[:reg.size],   reg,   label="Regularizer", linestyle=":", color="gray", linewidth=1.5)
        ax.set_ylabel("Objective Value"); ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9, ncol=3)
        ax.set_title("QP Objective Component Breakdown", fontsize=11)
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"[SAVED] {fname}")

    # ========================= QP PERFORMANCE ============================
    def plot_qp_performance(self, title_prefix="QP_Performance"):
        ts = np.asarray(self.npz.get("t_epoch", []))
        if ts.size == 0:
            print("[plot_qp_performance] No time axis.")
            return
        t = ts - ts[0]
        solve_ms = np.asarray(self.npz.get("solver_time", np.zeros(0))) * 1000.0
        if solve_ms.size == 0:
            print("[plot_qp_performance] No solver_time array.")
            return

        loop_diff = np.diff(ts)
        loop_diff = np.insert(loop_diff, 0, loop_diff[0] if loop_diff.size else 0.0)
        loop_hz = 1.0 / np.maximum(loop_diff, 1e-9)

        # simple rolling mean (window=10)
        if loop_hz.size >= 10:
            w = np.ones(10) / 10.0
            loop_hz_smooth = np.convolve(loop_hz, w, mode="same")
        else:
            loop_hz_smooth = loop_hz

        target_Hz = float(self.npz.get("meta__Hz", np.nan))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{title_prefix}", fontsize=13, y=0.98)

        axes[0].plot(t[:solve_ms.size], solve_ms, label="Solver Time", color="orange")
        if target_Hz == target_Hz and target_Hz > 0:
            axes[0].axhline(y=1000.0/target_Hz, color='r', linestyle='--', alpha=0.5, label=f"{int(target_Hz)}Hz limit")
        axes[0].set_ylabel("Time [ms]"); axes[0].set_title("Solver Duration per Iteration")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="upper right")

        axes[1].plot(t[:loop_hz_smooth.size], loop_hz_smooth, label="Loop Hz (Smoothed)", color="blue")
        if target_Hz == target_Hz and target_Hz > 0:
            axes[1].axhline(y=target_Hz, color='k', linestyle='--', alpha=0.5, label=f"Target {int(target_Hz)}Hz")
        axes[1].set_ylabel("Frequency [Hz]"); axes[1].set_xlabel("Time [s]")
        axes[1].set_title("Control Loop Frequency")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right")

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        fname = f"plots/{title_prefix.lower()}_{ts_str}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"[SAVED] {fname}")

    # ========================= FORCE PROFILE =============================
    def plot_force_profile(self, title_prefix="DualAdmittanceAccel2Vel"):
        data_t = np.asarray(self.npz.get("t_epoch", []))
        if data_t.size == 0:
            print("[plot_force_profile] No time axis.")
            return
        t = data_t - data_t[0]

        F_n_L = np.asarray(self.npz.get("F_n_L", np.zeros(0)))
        F_n_R = np.asarray(self.npz.get("F_n_R", np.zeros(0)))
        F_ref = np.asarray(self.npz.get("ref_force", np.zeros(0)))
        x_n_L = np.asarray(self.npz.get("x_n_L", np.zeros(0)))
        x_n_R = np.asarray(self.npz.get("x_n_R", np.zeros(0)))
        v_n_L = np.asarray(self.npz.get("v_n_L", np.zeros(0)))
        v_n_R = np.asarray(self.npz.get("v_n_R", np.zeros(0)))

        if F_n_L.size == 0 and F_n_R.size == 0:
            print("[plot_force_profile] No force/admittance arrays.")
            return

        fig1, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig1.suptitle(f"{title_prefix} – Normal / Admittance Metrics")

        ax = axes[0]
        if F_n_L.size: ax.plot(t[:F_n_L.size], F_n_L, label="F_n_L")
        if F_n_R.size: ax.plot(t[:F_n_R.size], F_n_R, label="F_n_R")
        if F_ref.size: ax.plot(t[:F_ref.size], F_ref, "--", label="F_ref")
        ax.set_ylabel("Normal force [N]"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[1]
        if x_n_L.size: ax.plot(t[:x_n_L.size], x_n_L, label="x_n_L (d_L)")
        if x_n_R.size: ax.plot(t[:x_n_R.size], x_n_R, label="x_n_R (d_R)")
        ax.set_ylabel("Distance along -n [m]"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[2]
        if v_n_L.size: ax.plot(t[:v_n_L.size], v_n_L, label="v_n_L")
        if v_n_R.size: ax.plot(t[:v_n_R.size], v_n_R, label="v_n_R")
        ax.set_ylabel("v_n [m/s]"); ax.set_xlabel("time [s]"); ax.grid(True, alpha=0.3); ax.legend()

        fig1.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Base-frame extras (only if present)
        p_L = np.asarray(self.npz.get("p_L", np.zeros((0,3))))
        p_R = np.asarray(self.npz.get("p_R", np.zeros((0,3))))
        grasp_L = np.asarray(self.npz.get("grasp_L", np.zeros((0,3))))
        grasp_R = np.asarray(self.npz.get("grasp_R", np.zeros((0,3))))
        d_L = np.asarray(self.npz.get("d_L", np.zeros(0)))
        d_R = np.asarray(self.npz.get("d_R", np.zeros(0)))
        v_tcp_L = np.asarray(self.npz.get("v_tcp_L", np.zeros((0,6))))
        v_tcp_R = np.asarray(self.npz.get("v_tcp_R", np.zeros((0,6))))
        v_des_L = np.asarray(self.npz.get("v_des_L", np.zeros((0,6))))
        v_des_R = np.asarray(self.npz.get("v_des_R", np.zeros((0,6))))

        fig2, axes2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig2.suptitle(f"{title_prefix} – Base-frame Metrics")

        labels_xyz = ["x","y","z"]
        for j in range(2):
            ax = axes2[0, j]
            if j == 0:
                p, g, side = p_L, grasp_L, "LEFT"
            else:
                p, g, side = p_R, grasp_R, "RIGHT"
            if p.shape[0]:
                for k in range(3):
                    ax.plot(t[:p.shape[0]], p[:,k], label=f"p_{labels_xyz[k]}")
                    if g.shape[0]:
                        ax.plot(t[:g.shape[0]], g[:,k], "--", label=f"grasp_{labels_xyz[k]}" if k == 0 else None)
            ax.set_ylabel(f"{side} TCP / grasp [m]"); ax.grid(True, alpha=0.3)
            if j == 0: ax.legend(ncol=3, fontsize=8)

        ax = axes2[1,0]
        if d_L.size: ax.plot(t[:d_L.size], d_L, label="d_L (along -n)")
        ax.set_ylabel("d_L [m]"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes2[1,1]
        if d_R.size: ax.plot(t[:d_R.size], d_R, label="d_R (along -n)")
        ax.set_ylabel("d_R [m]"); ax.grid(True, alpha=0.3); ax.legend()

        def _norm3(a): 
            return np.linalg.norm(a[:,:3], axis=1) if a.shape[0] else np.zeros(0)
        v_tcp_L_norm = _norm3(v_tcp_L)
        v_tcp_R_norm = _norm3(v_tcp_R)
        v_des_L_norm = _norm3(v_des_L)
        v_des_R_norm = _norm3(v_des_R)

        ax = axes2[2,0]
        if v_tcp_L_norm.size: ax.plot(t[:v_tcp_L_norm.size], v_tcp_L_norm, label="|v_tcp_L| meas")
        if v_des_L_norm.size: ax.plot(t[:v_des_L_norm.size], v_des_L_norm, "--", label="|v_des_L| cmd")
        ax.set_ylabel("LEFT |v| [m/s]"); ax.set_xlabel("time [s]"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes2[2,1]
        if v_tcp_R_norm.size: ax.plot(t[:v_tcp_R_norm.size], v_tcp_R_norm, label="|v_tcp_R| meas")
        if v_des_R_norm.size: ax.plot(t[:v_des_R_norm.size], v_des_R_norm, "--", label="|v_des_R| cmd")
        ax.set_ylabel("RIGHT |v| [m/s]"); ax.set_xlabel("time [s]"); ax.grid(True, alpha=0.3); ax.legend()

        os.makedirs("plots", exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        f1 = os.path.join("plots", f"{title_prefix}_normal_{ts_str}.png")
        f2 = os.path.join("plots", f"{title_prefix}_base_{ts_str}.png")
        fig1.savefig(f1, dpi=200); fig2.savefig(f2, dpi=200)
        plt.close(fig1); plt.close(fig2)
        print(f"[SAVED] {f1}\n[SAVED] {f2}")

    # ======================= BOX PATH (meas vs ref) ======================
    def plot_box_path_QP(self, title_prefix="Box_Meas_vs_Ref"):
        # Try left robot stream first (you record to robot_L.box_data in run())
        pos_meas = None; R_meas = None
        for side in ("L","R"):
            pkey = f"box__{side}_positions"; rkey = f"box__{side}_rotmats"
            if pkey in self.npz and rkey in self.npz:
                pos_meas = np.asarray(self.npz[pkey]); R_meas = np.asarray(self.npz[rkey])
                break
        if pos_meas is None or pos_meas.shape[0] == 0:
            print("[plot_box_path] No measured box pose arrays found.")
            return

        # Reference: either precomputed in saver or reconstruct from planner + frame
        has_ref = False
        if "box_ref_positions" in self.npz and "box_ref_rotmats" in self.npz:
            pos_ref = np.asarray(self.npz["box_ref_positions"])
            R_ref   = np.asarray(self.npz["box_ref_rotmats"])
            has_ref = True
        elif ("traj__rotation_matrices" in self.npz and
              "meta__grasping_point_L_world" in self.npz):  # frame might not be saved; try best-effort
            # Not enough info to rebuild perfectly without frame; skip instead of guessing.
            has_ref = False

        # Time
        T = np.asarray(self.npz.get("t_epoch", []))
        T = (T - T[0]) if T.size else np.arange(len(pos_meas), dtype=float)
        if has_ref:
            T_ref = np.linspace(T[0], T[min(len(T)-1, len(T)-1)], num=len(pos_ref))
        else:
            T_ref = None

        # rot → rpy
        def rot_to_rpy(Rstack):
            try:
                return Rotation.from_matrix(Rstack).as_euler('xyz', degrees=True)
            except Exception:
                eul = np.zeros((len(Rstack),3))
                r11, r21, r31 = Rstack[:,0,0], Rstack[:,1,0], Rstack[:,2,0]
                r32, r33 = Rstack[:,2,1], Rstack[:,2,2]
                eul[:,0] = np.degrees(np.arctan2(r32, r33))
                eul[:,1] = np.degrees(np.arcsin(-r31))
                eul[:,2] = np.degrees(np.arctan2(r21, r11))
                return eul

        rpy_meas = rot_to_rpy(R_meas)
        rpy_ref  = rot_to_rpy(R_ref) if has_ref else None

        os.makedirs("plots", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"plots/{title_prefix.lower()}_{stamp}.png"

        plt.figure(figsize=(14, 8))
        colors = ["red","green","blue"]; labels = ["X","Y","Z"]

        # Position vs time
        plt.subplot(2,1,1)
        for i,c in enumerate(colors):
            plt.plot(T[:pos_meas.shape[0]], pos_meas[:,i], label=f"meas {labels[i]}", lw=2, color=c)
            if has_ref:
                plt.plot(T_ref, pos_ref[:,i], "--", lw=2, color=c, alpha=0.7, label=f"ref {labels[i]}")
        plt.title("Box Position vs Time"); plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
        plt.legend(); plt.grid(True, alpha=0.3)

        # Orientation vs time (RPY)
        plt.subplot(2,1,2)
        rpy_names = ["Roll (deg)","Pitch (deg)","Yaw (deg)"]
        for i,c in enumerate(colors):
            plt.plot(T[:rpy_meas.shape[0]], rpy_meas[:,i], label=f"meas {rpy_names[i]}", lw=2, color=c)
            if has_ref:
                plt.plot(T_ref, rpy_ref[:,i], "--", lw=2, color=c, alpha=0.7, label=f"ref {rpy_names[i]}")
        plt.title("Box Orientation vs Time (RPY)"); plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(); print(f"[SAVED] {fname}")


if __name__ == "__main__":
    post_process = PostProcess("experiments/QP/logs/pick_and_place_0.5v_0.25a_0.5w_0.25B_QP_vention_4.4kg_20251209-162847.npz")
    #post_process = PostProcess("experiments/PID_ff/logs/angular/angular_PID_ff_bw_20251208-184626_R.npz")
    #post_process.plot_pid_tracking()
    #post_process.plot_data3D()
    post_process.plot_taskspace("L", title_prefix="DualArmQP_Taskspace_L")
    # post_process.plot_jointspace("L", title_prefix="DualArmQP_Jointspace_L")
    # post_process.plot_qp_performance(title_prefix="DualArmQP_QP_Performance")
    # post_process.plot_qp_objective(title_prefix="DualArmQP_QP_Objective")
    # post_process.plot_force_profile(title_prefix="DualArmQP_Force_Profile")
    # post_process.plot_box_path_QP(title_prefix="DualArmQP_Box_Meas_vs_Ref")

    #post_process.evaluate_PID("experiments/PID_ff/logs/linear_PID_ff_bw_20251208-144052_L.npz", "experiments/PID_ff/logs/linear_PID_ff_bw_20251208-144052_R.npz")
    #post_process.evaluate_PID("experiments/PID_ff/logs/angular/angular_PID_ff_bw_20251208-184626_R.npz", "experiments/PID_ff/logs/angular/angular_PID_ff_bw_20251208-184626_L.npz")
    #post_process.evaluate_QP("experiments/QP/logs/linear_QP_bw_20251208-143924.npz")
    #post_process.evaluate_QP2("experiments/QP/logs/angular_QP_bw_20251209-104314.npz", 5.0, 50)