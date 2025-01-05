import math
import numpy as np
import matplotlib.pyplot as plt


from agent import Agent
from slove_ilqr import Ilqr

# main function test cilqr
if __name__ == '__main__':

    max_iter = 20
    horizon = 30
    dt = 0.2

    #state contains [x y v theta]
    #input contains [a dtheta]
    ego = Agent(0, 0, 3, 0, 0, 0)
    game_agent = Agent(20, 0, 3, math.pi, 0, 0)

    game_agent_traj = game_agent.init_game_agent_traj(horizon,dt)


    u0 = np.zeros([2,horizon])
    ilqr = Ilqr(max_iter,horizon,dt)
    init_traj = ilqr.optimize(ego,game_agent_traj,u0)

    plt.scatter(init_traj[0,:],init_traj[1,:])
    plt.scatter(game_agent_traj[0,:], game_agent_traj[1,:])
    plt.show()
