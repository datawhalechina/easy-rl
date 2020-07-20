#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 10:01:09
@LastEditor: John
@LastEditTime: 2020-06-13 00:24:31
@Discription: 
@Environment: python 3.7.7
'''
'''
应该是没有收敛，但是pytorch官方教程的结果也差不多
'''
import gym
import torch

from screen_state import get_screen
from dqn import DQN
from plot import plot

import argparse

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--gamma", default=0.999, type=float) # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.9, type=float) # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay", default=200, type=float)

    parser.add_argument("--memory_capacity", default=10000, type=int,help="capacity of Replay Memory")

    parser.add_argument("--batch_size", default=128, type=int,help="batch size of memory sampling")
    parser.add_argument("--max_episodes", default=100, type=int)
    parser.add_argument("--max_steps", default=200, type=int)
    parser.add_argument("--target_update", default=4, type=int,help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()

    return config

if __name__ == "__main__":

    cfg = get_args()
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen(env,device)
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    agent = DQN(screen_height=screen_height, screen_width=screen_width,
             n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, memory_capacity=cfg.memory_capacity,batch_size=cfg.batch_size)

    rewards = []
    moving_average_rewards = []
    for i_episode in range(1,cfg.max_episodes+1):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        ep_reward = 0
        for t in range(1,cfg.max_steps+1):
            # Select and perform an action
            action = agent.select_action(state)     
            _, reward, done, _ = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward], device=device)
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)

            if done: break
            next_state = current_screen - last_screen

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.update()

        # Update the target network, copying all weights and biases in DQN
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %int(ep_reward), 'Explore: %.2f' % agent.epsilon)
        rewards.append(ep_reward)
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)

    import os
    import numpy as np
    output_path = os.path.dirname(__file__)+"/result/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(output_path+"rewards.npy", rewards)
    np.save(output_path+"moving_average_rewards.npy", moving_average_rewards)
    print('Complete！')
    plot(rewards)
    plot(moving_average_rewards,ylabel="moving_average_rewards")

    
