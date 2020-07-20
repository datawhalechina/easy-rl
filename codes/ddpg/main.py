#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
@LastEditTime: 2020-07-20 23:01:02
@Discription: 
@Environment: python 3.7.7
'''
import torch
import gym

from ddpg import DDPG
from env import NormalizedActions
from noise import OUNoise
from plot import plot

import argparse

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--gamma", default=0.99, type=float) # q-learning中的gamma
    parser.add_argument("--critic_lr", default=1e-3, type=float) # critic学习率
    parser.add_argument("--actor_lr", default=1e-4, type=float)

    parser.add_argument("--memory_capacity", default=10000, type=int,help="capacity of Replay Memory")

    parser.add_argument("--batch_size", default=128, type=int,help="batch size of memory sampling")
    parser.add_argument("--max_episodes", default=200, type=int)
    parser.add_argument("--max_steps", default=200, type=int)
    parser.add_argument("--target_update", default=4, type=int,help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()

    return config

if __name__ == "__main__":

    cfg = get_args()
    env = NormalizedActions(gym.make("Pendulum-v0"))
    
    # 增加action噪声
    ou_noise = OUNoise(env.action_space)
    
    n_states  = env.observation_space.shape[0] 
    n_actions = env.action_space.shape[0] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent=DDPG(n_states,n_actions,device="cpu", critic_lr=1e-3,
                 actor_lr=1e-4, gamma=0.99, soft_tau=1e-2, memory_capacity=100000, batch_size=128)

    rewards  = []
    moving_average_rewards = []
    for i_episode in range(1,cfg.max_episodes+1):
        state=env.reset()
        ou_noise.reset()
        ep_reward = 0
        for i_step in range(1,cfg.max_steps+1):
            action = agent.select_action(state)   
            action = ou_noise.get_action(action, i_step) # 即paper中的random process
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        print('Episode:', i_episode, ' Reward: %i' % int(ep_reward),)
        rewards.append(ep_reward)
        #
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
    print('Complete！')
    import os
    import numpy as np
    output_path = os.path.dirname(__file__)+"/result/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(output_path+"rewards.npy", rewards)
    np.save(output_path+"moving_average_rewards.npy", moving_average_rewards)
  
    plot(rewards)
    plot(moving_average_rewards,ylabel="moving_average_rewards")