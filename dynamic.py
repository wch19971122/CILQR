import math
import numpy as np

class Dynamic:


    length = 5.0
    width = 2.0
    lf = 1.8
    lr = 1.8
    cruise_speed = 4.0
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

    def get_A_matrix(self, state, input, horizon,dt,state_size,control_size):
        A = np.zeros([state_size, state_size, horizon])
        for i in range(horizon):
            v = state[2,i]
            v_dot = input[0,i]
            theta = state[3,i]
            A[:,:,i] = np.array([[1, 0, math.cos(theta) * dt, -(v * dt+ (v_dot * dt ** 2) / 2) * math.sin(theta)],
                          [0, 1, math.sin(theta) * dt, (v * dt + (v_dot * dt ** 2) / 2) * math.cos(theta)],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        return A

    def get_B_matrix(self, state, input, horizon,dt,state_size,control_size):
        B = np.zeros([state_size,control_size, horizon])
        for i in range(horizon):
            theta = state[3,i]
            B[:,:,i] = np.array([[dt ** 2 * math.cos(theta) / 2, 0],
                          [dt ** 2 * math.sin(theta) / 2, 0],
                          [dt , 0],
                          [0, dt]])
        return B
