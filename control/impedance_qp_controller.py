import numpy as np
import zmq
from control.pid_ff_controller import URForceController 

class JointOptimization():
    def __init__(self, robotL, robotR, Hz, trajectory):
        self.robotL = robotL
        self.robotR = robotR
        self.Hz = Hz
        self.trajectory = trajectory

    def run(self):

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


        