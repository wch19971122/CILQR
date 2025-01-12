import math
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from slove_ilqr import Ilqr


def get_box(x, y, theta):
    ego_cos = math.cos(theta)
    ego_sin = math.sin(theta)

    ego_box = np.array([[x + length / 2 * ego_cos - width / 2 * ego_sin,
                         x + length / 2 * ego_cos + width / 2 * ego_sin,
                         x - length / 2 * ego_cos + width / 2 * ego_sin,
                         x - length / 2 * ego_cos - width / 2 * ego_sin,
                         x + length / 2 * ego_cos - width / 2 * ego_sin],
                        [y + length / 2 * ego_sin + width / 2 * ego_cos,
                         y + length / 2 * ego_sin - width / 2 * ego_cos,
                         y - length / 2 * ego_sin - width / 2 * ego_cos,
                         y - length / 2 * ego_sin + width / 2 * ego_cos,
                         y + length / 2 * ego_sin + width / 2 * ego_cos]])
    return ego_box

# main function test cilqr
if __name__ == '__main__':

    max_iter = 20
    horizon = 30
    dt = 0.2
    length = 5
    width = 2

    #state contains [x y v theta]
    #input contains [a dtheta]
    ego = Agent(0, 0, 3, 0, 0, 0)
    game_agent = Agent(0, 0, 1, 0, 0, 0)

    reference_line = game_agent.init_game_agent_traj(horizon,dt)


    u0 = np.zeros([2,horizon])
    ilqr = Ilqr(max_iter,horizon,dt)
    init_traj,history_traj,valid_index = ilqr.optimize(ego,reference_line,u0)


    fig= plt.figure(figsize=(20,5))  # 生成画布
    plt.ion()  # 打开交互模式
    for index in range(horizon):
        fig.clf()  # 清空当前Figure对象
        plt.plot([-5,25],[3.5,3.5])
        plt.plot([-5,25],[-3.5,-3.5])
        plt.scatter(init_traj[0, :], init_traj[1, :], label='init_traj')
        plt.scatter(history_traj[0, :, valid_index], history_traj[1, :, valid_index], label='optimize_traj')
        plt.scatter(game_agent_traj[0, :], game_agent_traj[1, :], label='agent_traj')

        ego_box = get_box(history_traj[0,index,valid_index],history_traj[1,index,valid_index],history_traj[3,index,valid_index])
        agent_box = get_box(game_agent_traj[0,index],game_agent_traj[1,index],game_agent_traj[3,index])

        plt.plot(ego_box[0,:], ego_box[1,:], label='ego')
        plt.plot(agent_box[0,:], agent_box[1,:], label='agent')


        ego_state = history_traj[:, index, valid_index]
        game_state = game_agent_traj[:,index]

        # 暂停
        plt.pause(0.01)
        # plt.xlim([-5,25])
        # plt.ylim([-2,2])
        plt.axis('equal')
    # 关闭交互模式
    plt.ioff()
    plt.show()


