from control.pid_ff_controller import URForceController 

class URQPImpedanceController(URForceController):
    def __init__(self, ip):
        super().__init__(ip)


if __name__ == "__main__":
    robotL = URQPImpedanceController(
        "192.168.1.33"
    )
    print("Joints: ", robotL.get_joints())
    print("Mass matrix: ", robotL.get_mass_matrix())
    #print("Jacobian: {}", robotL.get_jacobian())
    # print("Jacobian derivative: {}", robotL.get_jacobian_derivative())
        