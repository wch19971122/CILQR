import math

class Dynamic:

    max_acc = 1.0
    min_acc = -1.0
    max_yaw_rate = 1.0
    min_yaw_rate = -1.0

    def __init__(self):
        self.x = 0

    def update(self, state, input, dt):
        x_next = state[0] + math.cos(state[3])*(state[2]*dt + dt**2*input[0]/2)
        y_next = state[1] + math.sin(state[3])*(state[2]*dt + dt**2*input[0]/2)
        v_next = state[2] + input[0]*dt
        theta_next = state[3] + input[1]*dt
        return [x_next, y_next, v_next, theta_next]

    def get_A_matrix(self, state, input):
        return 0

    def get_B_matrix(self, state, input):
        return 0
