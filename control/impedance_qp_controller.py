import numpy as np
import time
from control.pid_ff_controller import URForceController 

class JointOptimization():
    def __init__(self, robotL, robotR, Hz, trajectory):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz

        #references
        trajectory_npz=np.load(trajectory)
        self.position_ref = trajectory_npz['position']
        self.velocity_ref = trajectory_npz['linear_velocity']
        self.rotation_ref = trajectory_npz['rotation_matrices']

    def run(self):
        control_period = 1.0 / self.control_rate_hz
        trajectory_index = 0

        while not self.control_stop.is_set():
            loop_start = time.perf_counter()
            try:

                # =========================== Trajectory Updates ===========================

                trajectory_index += 1


                # --- Log ---
                self.control_data.append(
                    {

                    }
                )


            except Exception as e:
                print(f"Control error: {e}")
                try:
                    self.speedStop()
                except:
                    pass
                break

            # Keep loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, control_period - elapsed))

        # Clean stop
        try:
            self.speedStop()
        except:
            pass

class URImpedanceController(URForceController):
    def __init__(self, ip, K):
        super().__init__(ip)
        self.K = K
        self.D = np.zeros((6, 6))

        self.J = self.get_jacobian()
        self.Jdot =  self.get_jacobian_derivative()
        self.M = self.get_mass_matrix()
        self.Lambda= np.linalg.inv(self.J @ np.linalg.inv(self.M) @ self.J.T)

        self.cotrol_data = []   

    def get_D(self, K, Lambda):
        D = np.sqrt(Lambda) * np.sqrt(K) + np.sqrt(K) * np.sqrt(Lambda)
        return D
    
    def get_wrench_desired(self, D, K, position, velocity, rotation, position_ref, velocity_ref, rotation_ref,):
        wrench = D @ (velocity_ref - velocity) + K @ (position_ref - position)