import math

import numpy as np
from encodings.punycode import selective_find
from mpmath import zeros
from numpy.f2py.capi_maps import using_newcore
from sympy import false
from sympy.physics.control import forward_diff
from sympy.polys.subresultants_qq_zz import backward_eye
from sympy.utilities.codegen import JuliaCodeGen

from dynamic import Dynamic


class Ilqr:
    def __init__(self, max_iter, horizon, dt):
        self.max_iter = max_iter
        self.control_size = 2
        self.state_size = 4
        self.q1 = 1
        self.q2 = 1
        self.control_cost = np.array([[1, 0],
                                      [0, 1]])
        self.state_cost = np.array([[0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0]])
        self.horizon = horizon
        self.dt = dt
        self.dynamic = Dynamic()

    def optimize(self, ego, reference_line, u0):
        init_traj = self.get_init_traj(ego, u0)
        pre_traj = init_traj
        pre_u = u0

        lamda = np.ones([4, self.horizon])
        mu = 1

        J_old = self.get_al_cost(pre_traj, pre_u, reference_line, lamda, mu)

        history_traj = np.zeros([self.state_size, self.horizon + 1, self.max_iter])

        converge = false
        iter = 0
        while not converge:
            if iter > self.max_iter:
                print("over max iter count!")
                break

            iter = iter + 1

            k, K = self.backward_pass(pre_traj, reference_line, pre_u, lamda, mu)
            x_new, u_new = self.forward_pass(pre_traj, pre_u, k, K)

            J_new = self.get_cost(x_new, u_new)

            if J_new < J_old:
                pre_traj = x_new
                pre_u = u_new
                history_traj[:, :, i] = pre_traj
                lamda = lamda / 3
                valid_index = i
                if np.abs(J_new - J_old) < 0.0001:
                    break
            else:
                lamda = 2 * lamda
                if lamda > 5:
                    break

            J_old = J_new

        return init_traj, history_traj, valid_index

    def get_init_traj(self, ego, u0):
        traj = np.zeros([4, self.horizon + 1])
        traj[:, 0] = [ego.x, ego.y, ego.v, ego.theta]
        for i in range(self.horizon):
            traj[:, i + 1] = self.dynamic.update(traj[:, i], u0[:, i], self.dt)
        return traj

    def forward_pass(self, x, u, k, K):
        x_new = np.zeros([self.state_size, self.horizon + 1])
        x_new[:, 0] = x[:, 0]
        u_new = np.zeros([self.control_size, self.horizon])
        for i in range(self.horizon):
            u_new[:, i] = u[:, i] + k[:, i] + K[:, :, i] @ (x_new[:, i] - x[:, i])
            x_new[:, i + 1] = self.dynamic.update(x_new[:, i], u_new[:, i], self.dt)
        return x_new, u_new

    def backward_pass(self, pre_traj, reference_line, u0, lamda, mu):
        lu, luu, lx, lxx, lux, lxu = self.get_al_cost_drivative(pre_traj, reference_line, u0, lamda, mu)
        fx, fu = self.get_f_drivative(pre_traj, u0)

        pN = lx[:, -1]
        PN = lxx[:, :, -1]
        k = np.zeros([self.control_size, self.horizon])
        K = np.zeros([self.control_size, self.state_size, self.horizon])
        rho = 0.1
        for i in range(self.horizon - 1, -1, -1):
            Qx = lx[:, i] + pN.T @ fx[:, :, i]
            Qu = lu[:, i] + pN.T @ fu[:, :, i]
            Qxx = lxx[:, :, i] + fx[:, :, i].T @ PN @ fx[:, :, i]
            Quu = luu[:, :, i] + fu[:, :, i].T @ PN @ fu[:, :, i]
            Qxu = lxu[:, :, i] = fx[:, :, i].T @ PN @ fu[:, :, i]
            Qux = lux[:, :, i] = fu[:, :, i].T @ PN @ fx[:, :, i]

            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Quu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamda
            Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))

            # Quu_eigval = np.linalg.eigvals(Quu)
            # while(not np.all(Quu_eigval) > 0):
            #     Quu + rho*np.ones([self.control_size,self.control_size])
            #
            # Q_uu_inv = np.linalg.inv(Quu)

            K[:, :, i] = -Q_uu_inv @ Qux
            k[:, i] = -Q_uu_inv @ Qu

            pN = Qx + K[:, :, i].T @ Quu @ k[:, i] + Qxu @ k[:, i] + K[:, :, i].T @ Qu
            PN = Qxx + K[:, :, i].T @ Quu @ K[:, :, i] + Qxu @ K[:, :, i] + K[:, :, i].T @ Qux

        return k, K

    def get_f_drivative(self, pre_traj, u0):
        A = self.dynamic.get_A_matrix(pre_traj, u0, self.horizon, self.dt, self.state_size, self.control_size)
        B = self.dynamic.get_B_matrix(pre_traj, u0, self.horizon, self.dt, self.state_size, self.control_size)
        return A, B

    def get_cost_drivative(self, pre_traj, game_agent_traj, u0):
        lu, luu = self.get_control_cost_drivative(pre_traj, game_agent_traj, u0)
        lx, lxx = self.get_state_cost_drivative(pre_traj, game_agent_traj)
        lux = np.zeros((self.control_size, self.state_size, self.horizon))
        lxu = np.zeros((self.state_size, self.control_size, self.horizon))
        return lu, luu, lx, lxx, lux, lxu

    def get_control_cost_drivative(self, pre_traj, game_agent_traj, u0):
        lu = np.zeros([self.control_size, self.horizon])
        luu = np.zeros([self.control_size, self.control_size, self.horizon])
        P1 = np.array([1, 0])
        P2 = np.array([0, 1])

        for i in range(self.horizon):
            # max acc
            max_acc_b, max_acc_db, max_acc_ddb = self.get_barrier_function(self.q1, self.q2,
                                                                           u0[:, i].T @ P1 - self.dynamic.max_acc, P1)

            # min acc
            min_acc_b, min_acc_db, min_acc_ddb = self.get_barrier_function(self.q1, self.q2,
                                                                           self.dynamic.min_acc - u0[:, i].T @ P1, -P1)

            # max yaw_rate
            max_yawRate_b, max_yawRate_db, max_yawRate_ddb = self.get_barrier_function(self.q1, self.q2, u0[:,
                                                                                                         i].T @ P2 - self.dynamic.max_yaw_rate,
                                                                                       P2)

            # min yaw_rate
            min_yawRate_b, min_yawRate_db, min_yawRate_ddb = self.get_barrier_function(self.q1, self.q2,
                                                                                       self.dynamic.min_yaw_rate - u0[:,
                                                                                                                   i].T @ P2,
                                                                                       -P2)

            lu[:, i] = max_acc_db + max_acc_ddb + min_acc_db + min_acc_ddb + 2 * u0[:, i].T @ self.control_cost
            luu[:, :, i] = max_acc_ddb + min_acc_ddb + min_yawRate_db + min_yawRate_ddb + 2 * self.control_cost

        return lu, luu

    def get_state_cost_drivative(self, pre_traj, game_agent_traj):
        lx = np.zeros([self.state_size, self.horizon])
        lxx = np.zeros([self.state_size, self.state_size, self.horizon])

        for i in range(self.horizon):
            lx_i = 2 * (pre_traj[:, i] - [0, 0, self.dynamic.cruise_speed, 0]).T @ self.state_cost
            lxx_i = 2 * self.state_cost

            obs_db, obs_ddb = self.get_obstacle_cost_derivatives(game_agent_traj[:, i], pre_traj[:, i])

            lx_i.squeeze()
            obs_db = np.squeeze(obs_db)

            lx[:, i] = lx_i + obs_db
            lxx[:, :, i] = lxx_i + obs_ddb
        return lx, lxx

    def get_obstacle_cost_derivatives(self, agent_state, ego_state):

        a = self.dynamic.length
        b = self.dynamic.width

        P1 = np.diag([1 / a ** 2, 1 / b ** 2, 0, 0])

        theta = agent_state[3]
        theta_ego = ego_state[3]

        transformation_matrix = np.array([[math.cos(theta), math.sin(theta), 0, 0],
                                          [-math.sin(theta), math.cos(theta), 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]])

        ego_front = ego_state + np.array(
            [math.cos(theta_ego) * self.dynamic.lf, math.sin(theta_ego) * self.dynamic.lf, 0, 0])
        diff = (transformation_matrix @ (ego_front - agent_state)).reshape(-1, 1)  # (x- xo)
        c = 1 - diff.T @ P1 @ diff  # Transform into a constraint function
        c_dot = -2 * P1 @ diff
        b_f, b_dot_f, b_ddot_f = self.get_barrier_function(10, self.q2, c, c_dot)

        ego_rear = ego_state - np.array(
            [math.cos(theta_ego) * self.dynamic.lr, math.sin(theta_ego) * self.dynamic.lr, 0, 0])
        diff = (transformation_matrix @ (ego_rear - agent_state)).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_r, b_dot_r, b_ddot_r = self.get_barrier_function(10, self.q2, c, c_dot)

        return b_dot_f + b_dot_r, b_ddot_f + b_ddot_r

    def get_barrier_function(self, q1, q2, c, dc):
        x = q1 * np.exp(q2 * c)
        dx = q1 * q2 * np.exp(q2 * c) * dc
        ddx = q1 * (q2 ** 2) * np.exp(q2 * c) * dc * dc.T
        return x, dx, ddx

    def get_cost(self, x, u):
        J = 0
        for i in range(self.horizon):
            state_diff = x[:, i] - np.array([0, 0, self.dynamic.cruise_speed, 0])
            state_cost = state_diff.T @ self.state_cost @ state_diff

            control_cost = u[:, i].T @ self.control_cost @ u[:, i]

            J += state_cost + control_cost
        return J

    def get_al_cost(self, x, u, reference_line, lamda, mu):
        J = 0
        for i in range(self.horizon):
            state_diff = x[:, i] - reference_line[:, i]
            state_cost = state_diff.T @ self.state_cost @ state_diff
            control_cost = u[:, i].T @ self.control_cost @ u[:, i]

            control_al_cost = self.get_al_control_cost(u[:, i], lamda[:, i], mu, self.dynamic.max_acc,
                                                       self.dynamic.max_yaw_rate)
            J += state_cost + control_cost + control_al_cost
        return J

    def get_al_control_cost(self, u, lamda, mu, max_acc, max_yaw_rate):

        J_control = mu / 2 * (pow(max(lamda[0] / mu + (u[0] - max_acc), 0), 2) - pow(lamda[0] / mu, 2) +
                              pow(max(lamda[1] / mu + (-u[0] - max_acc), 0), 2) - pow(lamda[1] / mu, 2) +
                              pow(max(lamda[2] / mu + (u[1] - max_acc), 0), 2) - pow(lamda[2] / mu, 2) +
                              pow(max(lamda[3] / mu + (-u[1] - max_acc), 0), 2) - pow(lamda[3] / mu, 2))
        return J_control

    def get_al_cost_drivative(self, pre_traj, reference_line, u0, lamda, mu):
        Jx = np.zeros([self.state_size, self.horizon])
        Ju = np.zeros([self.control_size, self.horizon])
        Jxx = np.zeros([self.state_size, self.state_size, self.horizon])
        Juu = np.zeros([self.control_size, self.control_size, self.horizon])
        Jxu = np.zeros([self.state_size, self.control_size, self.horizon])
        Jux = np.zeros([self.control_size, self.state_size, self.horizon])

        # 假设u没有参考值，则Ju，Juu，Jux,Jxu
        for i in range(self.horizon):
            Jx[:, i] = (pre_traj[:, i] - reference_line[:, i]).T @ self.state_cost
            Jxx[:, :, i] = self.state_cost

        return Jx, Ju, Jxx, Juu, Jxu, Jux
