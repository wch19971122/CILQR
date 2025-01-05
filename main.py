import math
import numpy as np
import matplotlib.pyplot as plt


from agent import Agent

# main function test cilqr
if __name__ == '__main__':

    horizon = 30
    dt = 0.2

    #state contains [x y v theta]
    #input contains [a dtheta]
    ego = Agent(0, 0, 2, 0, 0, 0)
    gameAgent = Agent(20, 0, 3, math.pi, 0, 0)

    gameA_agent_traj = gameAgent.init_game_agent_traj(horizon,dt)

    plt.scatter(gameA_agent_traj[0,:], gameA_agent_traj[1,:])
    plt.show()
