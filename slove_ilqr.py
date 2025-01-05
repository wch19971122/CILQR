import math

import numpy as np
from encodings.punycode import selective_find
from sympy.polys.subresultants_qq_zz import backward_eye

from dynamic import Dynamic

class Ilqr:
    def __init__(self, max_iter ,horizon,dt):
        self.max_iter = max_iter
        self.control_size = 2
        self.state_size = 4
        self.q1 = 1
        self.q2 = 1
        self.control_cost = np.array([[1,0],
                                     [0,1]])
        self.horizon = horizon
        self.dt = dt
        self.dynamic = Dynamic()

    def optimize(self, ego, game_agent_traj, u0):
        init_traj =  self.get_init_traj(ego,u0)

        for i in range(self.max_iter):
            pre_traj = init_traj
            pre_u0 = u0
            self.backward_pass(pre_traj,game_agent_traj,u0)


        return init_traj

    def get_init_traj(self, ego, u0):
        traj = np.zeros([4,self.horizon+1])
        traj[:,0] = [ego.x, ego.y, ego.v, ego.theta]
        for i in range(self.horizon):
            traj[:,i+1] = self.dynamic.update(traj[:,i], u0[:,i], self.dt)
        return traj


    def forward_pass(self, x, u):
        return x

    def backward_pass(self, pre_traj, game_agent_traj,u0):
        lx,lxx,lu,luu,lux,lxu = self.get_cost_drivative(pre_traj, game_agent_traj,u0)

        return 0

    def get_cost_drivative(self,pre_traj, game_agent_traj,u0):
        lu, luu = self.get_control_cost_drivative(pre_traj,game_agent_traj,u0)
        lx, lxx = self.get_state_cost_drivative(pre_traj,game_agent_traj)
        return 0,0,0,0,0,0

    def get_control_cost_drivative(self, pre_traj,game_agent_traj,u0):
        lu = np.zeros([self.control_size,self.horizon])
        luu = np.zeros([self.control_size,self.control_size,self.horizon])
        P1 = np.array([1,0])
        P2 = np.array([0,1])

        for i in range(self.horizon):
            #max acc
            max_acc_b, max_acc_db, max_acc_ddb = self.get_barrier_function(self.q1, self.q2, u0[:,i].T @ P1 - self.dynamic.max_acc, P1)

            #min acc
            min_acc_b, min_acc_db, min_acc_ddb = self.get_barrier_function(self.q1, self.q2, self.dynamic.min_acc - u0[:,i].T @ P1, -P1)

            #max yaw_rate
            max_yawRate_b,max_yawRate_db,max_yawRate_ddb = self.get_barrier_function(self.q1, self.q2, u0[:,i].T @ P2 - self.dynamic.max_yaw_rate, P2)

            # min yaw_rate
            min_yawRate_b, min_yawRate_db, min_yawRate_ddb = self.get_barrier_function(self.q1, self.q2, self.dynamic.min_yaw_rate- u0[:,i].T @ P2,-P2)

            lu = max_acc_db + max_acc_ddb + min_acc_db + min_acc_ddb + 2*u0[:,i].T @ self.control_cost
            luu = max_acc_ddb + min_acc_ddb + min_yawRate_db + min_yawRate_ddb + 2*self.control_cost
            return lu, luu

        return 0

    def get_state_cost_drivative(self,pre_traj,game_agent_traj):
        return 0, 0


    def get_barrier_function(self,q1,q2,c,dc):
        x = q1*np.exp(q2*c)
        dx = q1*q2*np.exp(q2*c)*dc
        ddx = q1*(q2**2)*np.exp(q2*c)*dc*(dc.T)
        return x, dx, ddx







