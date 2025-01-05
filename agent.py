import numpy as np
import sympy
from sympy import atanh


class Agent:
    def __init__(self,x,y,v,theta,a,dtheta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.a = a
        self.dtheta = dtheta

    def init_game_agent_traj(self, horizon, dt,):
        # 定义符号变量
        x, a, b, c, d = sympy.symbols('x a b c d')

        # 输入起点和终点坐标以及起点处的一阶导数
        x1 = 20  # 起点横坐标，可替换为实际值
        y1 = 0  # 起点纵坐标，可替换为实际值
        x2 = -2  # 终点横坐标，可替换为实际值
        y2 = 3  # 终点纵坐标，可替换为实际值
        y1_prime = 0  # 起点处一阶导数，可替换为实际值
        y2_prime = 0

        # 根据三次多项式及其导数与已知条件建立方程组
        eq1 = sympy.Eq(a * x1 ** 3 + b * x1 ** 2 + c * x1 + d, y1)
        eq2 = sympy.Eq(a * x2 ** 3 + b * x2 ** 2 + c * x2 + d, y2)
        eq3 = sympy.Eq(3 * a * x1 ** 2 + 2 * b * x1 + c, y1_prime)
        # 再补充一个额外条件，通常为连续性条件等，这里可以用函数在终点处的二阶导数为0（一种常见设定，可根据实际改）
        eq4 = sympy.Eq(3 * a * x2 ** 2 + 2 * b * x2 + c, y2_prime)

        # 解方程组
        solution = sympy.solve([eq1, eq2, eq3, eq4], [a, b, c, d])
        a_value = float(solution[a])
        b_value = float(solution[b])
        c_value = float(solution[c])
        d_value = float(solution[d])

        game_agent_traj = np.zeros([4, 30])

        v_ave = abs(x2-x1)/(horizon*dt)

        for i in range(horizon):
            x_cur = self.x - i*dt*v_ave
            y_cur = a_value*x_cur**3 + b_value*x_cur**2 + c_value*x_cur + d_value
            v = v_ave
            theta = atanh((3*a_value*x_cur**2 + 2 * b_value*x_cur + c_value)/v_ave)
            game_agent_traj[:,i] = [x_cur,y_cur,v,theta]

        return game_agent_traj



