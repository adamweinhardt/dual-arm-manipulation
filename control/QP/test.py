#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import threading
import queue
import time
from numpy import pi
from scipy.spatial.transform import Rotation as RR
# Assuming this module provides the necessary URForceController base class
from control.UR.ur_controller import URController
from scipy.linalg import sqrtm
import pinocchio as pin
from rtde_control import RTDEControlInterface # Included for the main execution block
from rtde_receive import RTDEReceiveInterface
from scipy.linalg import norm

def rvec_to_rotmat(rvec):
    return RR.from_rotvec(rvec).as_matrix()

def rotmat_to_rvec(R):
    return RR.from_matrix(R).as_rotvec()

# ===========================
# BASE CLASS (Your Pinocchio Implementation)
# ===========================

class URImpedanceController(URController):
    """
    Thin wrapper to expose kinematics/dynamics via Pinocchio.
    K is the impedance stiffness (6x6, SPD).
    """

    def __init__(self, ip, K):
        # Assumes URForceController provides __init__ which sets up RTDE connections
        super().__init__(ip)
        self.K = np.asarray(K, dtype=float)
        # Pinocchio setup
        try:
            self.pin_model = pin.buildModelFromUrdf("ur5/UR5e.urdf")
            self.pin_data = self.pin_model.createData()
            self.pin_frame_id = self.pin_model.getFrameId("tool0")
        except Exception:
            print("[Warning] Pinocchio URDF could not be loaded. Pinocchio methods will fail.")
            self.pin_model = None


    # state
    def get_q(self):
        # Assuming self.rtde_receive is set up by URForceController
        return np.array(self.rtde_receive.getActualQ(), dtype=float)

    def get_qdot(self):
        # Assuming self.rtde_receive is set up by URForceController
        return np.array(self.rtde_receive.getActualQd(), dtype=float)

    # kinematics (Pinocchio versions, for context/backup)
    def get_J_pin(self, q):
        if not self.pin_model: return np.zeros((6, 6))
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        J = pin.getFrameJacobian(
            self.pin_model, self.pin_data, self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        # Assumes get_J_world is defined in URForceController or a mixin class it uses
        return self.get_J_world_pin(J)

    def get_Jdot_pin(self, q, v):
        if not self.pin_model: return np.zeros((6, 6))
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, v)
        dJ = pin.getFrameJacobianTimeVariation(
            self.pin_model, self.pin_data, self.pin_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        # Assumes get_J_world is defined in URForceController or a mixin class it uses
        return self.get_J_world_pin(dJ)

    # dynamics (Pinocchio version)
    def get_M_pin(self, q):
        if not self.pin_model: return np.zeros((6, 6))
        M = pin.crba(self.pin_model, self.pin_data, q)
        return np.asarray(0.5 * (M + M.T))

    # Impedance control calculations (common to both)
    def get_Lambda(self, J, M, tikhonov=1e-6):
        """
        Operational-space inertia: Λ ≈ (J M^{-1} Jᵀ + λI)^{-1}
        """
        Minv = np.linalg.inv(M)
        A = J @ Minv @ J.T
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        w_d = np.maximum(w + float(tikhonov), 1e-12)
        Lambda = V @ np.diag(1.0 / w_d) @ V.T
        return np.asarray(0.5 * (Lambda + Lambda.T))

    def get_D(self, K, Lambda):
        S_L = sqrtm(Lambda)
        S_K = sqrtm(K)
        D = S_L @ S_K + S_K @ S_L
        return np.real_if_close(0.5 * (D + D.T))

    def wrench_desired(self, K, D, e_p, e_r, e_v, e_w):
        X_err  = np.hstack((e_p, e_r))
        Xd_err = np.hstack((e_v, e_w))
        return D @ Xd_err + K @ X_err

# ===========================
# RTDE IMPLEMENTATION (Derived Class)
# ===========================

class URDynamicsWrapper(URImpedanceController):
    """
    Implements J, Jdot, and M using the UR RTDE API, overriding the Pinocchio 
    methods in the parent class to be used by the main QP loop.
    """

    # --- RTDE API Implementations (Overrides) ---

    def get_J(self, q):
        """Jacobian J via RTDE API, transformed to World frame."""
        # Assuming self.rtde_control is inherited and available
        J_flat = self.rtde_control.getJacobian(q.tolist())
        J_base = np.array(J_flat, dtype=float).reshape((6, 6))
        # Assumes get_J_world is inherited and available
        return self.get_J_world(J_base)

    def get_Jdot(self, q, v):
        """Jacobian Time Derivative Jdot via RTDE API, transformed to World frame."""
        # Assuming self.rtde_control is inherited and available
        Jdot_flat = self.rtde_control.getJacobianTimeDerivative(q.tolist(), v.tolist())
        Jdot_base = np.array(Jdot_flat, dtype=float).reshape((6, 6))
        # Assumes get_J_world is inherited and available
        return self.get_J_world(Jdot_base)

    def get_M(self, q):
        """Mass Matrix M via RTDE API (Joint Space)."""
        # Assuming self.rtde_control is inherited and available
        M_flat = self.rtde_control.getMassMatrix(q.tolist(), include_rotors_inertia=True)
        M = np.array(M_flat, dtype=float).reshape((6, 6))
        return np.asarray(0.5 * (M + M.T))

    def debug_matrices(self):
        """
        Calculates and prints the dynamics matrices from two sources:
        1. RTDE (Robot Kernel - the "measured" truth, J, Jdot, M)
        2. Pinocchio (Local PC Model - the "nominal" model, J_pin, Jdot_pin, M_pin)
        
        The function then prints the absolute error between them.
        """
        q = self.get_q()
        v = self.get_qdot()
        
        # Guard clause for RTDE access
        if not hasattr(self, 'rtde_control'):
            print("[FATAL] RTDE Control interface is not initialized.")
            return

        try:
            # ----------------------------------------------------
            # 1. RETRIEVE MATRICES (RTDE vs PINOCCHIO)
            # ----------------------------------------------------
            
            # RTDE (Measured)
            J_rtde = self.get_J(q)
            Jdot_rtde = self.get_Jdot(q, v)
            M_rtde = self.get_M(q)

            # Pinocchio (Nominal/Local) - Assumed available from parent class
            J_pin = self.get_J_pin(q)
            Jdot_pin = self.get_Jdot_pin(q, v)
            M_pin = self.get_M_pin(q)


            # ----------------------------------------------------
            # 2. CALCULATE ERRORS
            # ----------------------------------------------------
            
            # Error functions
            def max_abs_error(M1, M2):
                return np.max(np.abs(M1 - M2))
            
            def frobenius_norm(M1, M2):
                return norm(M1 - M2, 'fro')

            # Error Metrics
            J_err_max = max_abs_error(J_rtde, J_pin)
            J_err_norm = frobenius_norm(J_rtde, J_pin)
            
            Jdot_err_max = max_abs_error(Jdot_rtde, Jdot_pin)
            Jdot_err_norm = frobenius_norm(Jdot_rtde, Jdot_pin)
            
            M_err_max = max_abs_error(M_rtde, M_pin)
            M_err_norm = frobenius_norm(M_rtde, M_pin)


            # ----------------------------------------------------
            # 3. PRINT RESULTS
            # ----------------------------------------------------

            current_pose = self.rtde_receive.getActualTCPPose()

            print("\n" + "="*80)
            print(f" DYNAMICS COMPARISON DEBUG @ Q={np.round(q, 4)} ".center(80))
            print("="*80)
            print(f"Current TCP Pose (Base Frame): {np.round(current_pose, 4)}")
            print(f"Joint Velocity (Qdot): {np.round(v, 4)}")
            
            # --- JACOBIAN COMPARISON ---
            print("\n--- 1. JACOBIAN (J) [World Frame] ---")
            print(f"RTDE (Max Abs: {np.max(np.abs(J_rtde)):.4f} | Shape: {J_rtde.shape})")
            print(f"Linear J (RTDE):\n{np.round(J_rtde[:3,:], 4)}")
            
            print(f"\nPINOCCHIO (Max Abs: {np.max(np.abs(J_pin)):.4f})")
            print(f"Linear J (PIN):\n{np.round(J_pin[:3,:], 4)}")

            print("\n-- J COMPARISON METRICS --")
            print(f"MAX ABS ERROR: {J_err_max:.4e}")
            print(f"FROBENIUS NORM (Total Diff): {J_err_norm:.4e}")


            # --- JDOT COMPARISON ---
            print("\n--- 2. JACOBIAN TIME DERIVATIVE (JDOT) [World Frame] ---")
            print(f"RTDE (Max Abs: {np.max(np.abs(Jdot_rtde)):.4e} | Shape: {Jdot_rtde.shape})")
            print(f"Linear Jdot (RTDE):\n{np.round(Jdot_rtde[:3,:], 6)}")

            print(f"\nPINOCCHIO (Max Abs: {np.max(np.abs(Jdot_pin)):.4e})")
            print(f"Linear Jdot (PIN):\n{np.round(Jdot_pin[:3,:], 6)}")
            
            print("\n-- JDOT COMPARISON METRICS --")
            print(f"MAX ABS ERROR: {Jdot_err_max:.4e}")
            print(f"FROBENIUS NORM (Total Diff): {Jdot_err_norm:.4e}")
            
            
            # --- MASS MATRIX COMPARISON ---
            print("\n--- 3. MASS MATRIX (M) [Joint Space] ---")
            print(f"RTDE M Diagonal: {np.round(M_rtde.diagonal(), 4)}")
            print(f"RTDE M (Top-Left 3x3):\n{np.round(M_rtde[:3,:3], 4)}")

            print(f"\nPINOCCHIO M Diagonal: {np.round(M_pin.diagonal(), 4)}")
            print(f"PINOCCHIO M (Top-Left 3x3):\n{np.round(M_pin[:3,:3], 4)}")

            print("\n-- M COMPARISON METRICS --")
            print(f"MAX ABS ERROR: {M_err_max:.4e}")
            print(f"FROBENIUS NORM (Total Diff): {M_err_norm:.4e}")

            print("="*80 + "\n")

        except Exception as e:
            print(f"\n[ERROR] Failed to retrieve or compare dynamics matrices: {e}")
            print("Ensure the UR RTDE Interfaces are connected and Pinocchio is initialized.")

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    IP = "192.168.1.33" 
    K = np.diag([100.0] * 6)
    robot = None

    try:
        # NOTE: URForceController must be callable as BaseClass(ip, K)
        robot = URDynamicsWrapper(IP, K) 
        time.sleep(1)
        
        # --- COMMANDS (Assuming they are methods of URForceController) ---
        
        print(f"\n[START] Connecting to UR at {IP}. Executing debug sequence...")

        # --- STEP 1: Go Home and Debug ---
        print("\n--- STEP 1: Moving to Home Position ---")
        # Assuming robot.home_joints is defined in URForceController
        robot.moveJ(robot.home_joints)
        robot.wait_for_commands()
        time.sleep(2)

        print("\n--- DEBUG 1: State at Home (Qdot should be near zero) ---")
        robot.debug_matrices()
        
        # --- STEP 2: Go to Approach Position and Debug ---
        print("\n--- STEP 2: Moving to Approach Position ---")
        # Using go_to_approach() as requested
        robot.moveJ([
            -pi / 1.8,
            -pi / 1.8,
            pi / 1.8,
            -pi / 1.8,
            -pi / 1.8,
            pi,
        ])
        
        # Check and debug while moving (Jdot should be non-zero)
        robot.wait_for_commands() # Wait for non-blocking command to be processed
        time.sleep(1) 
        print("\n--- DEBUG 2: State while Moving (Jdot should be non-zero) ---")
        robot.debug_matrices()

        # Wait for the move to complete
        # NOTE: Using wait_until_done() is safer for moves, but wait_for_commands() 
        # is necessary to ensure the command queue is empty.
        robot.wait_for_commands() 
        time.sleep(1)

        print("\n--- DEBUG 3: State at Approach Position (Qdot near zero again) ---")
        robot.debug_matrices()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping control.")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        print("Please check UR RTDE setup and the availability of URForceController.")
    finally:
        if robot is not None:
             print("Stopping control and disconnecting.")
             # Attempt to stop robot motion before disconnecting
             try:
                 # Assuming rtde_control is accessible via self.rtde_control
                 if robot.rtde_control.isProgramRunning():
                      robot.rtde_control.stopScript()
             except:
                 pass
             robot.disconnect()