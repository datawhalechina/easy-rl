#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
@LastEditTime: 2020-07-20 23:02:16
@Discription: 
@Environment: python 3.7.7
'''
'''未完成
'''
import gym
import torch
from dqn import DQN
from plot import plot
import argparse
def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--gamma", default=0.99,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.95,
                        type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_decay", default=200, type=float)
    parser.add_argument("--policy_lr", default=0.01, type=float)
    parser.add_argument("--memory_capacity", default=1000,
                        type=int, help="capacity of Replay Memory")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--max_episodes", default=200, type=int)
    parser.add_argument("--max_steps", default=200, type=int)
    parser.add_argument("--target_update", default=2, type=int,
                        help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()

    return config


if __name__ == "__main__":

    cfg = get_args()
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0').unwrapped
    env.seed(1)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay,policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
    rewards = []
    moving_average_rewards = []
    for i_episode in range(1, cfg.max_episodes+1):
        # Initialize the environment and state
        state = env.reset()
        ep_reward = 0
        for t in range(1, cfg.max_steps+1):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # Store the transition in memory
            agent.memory.push(state,action,reward,next_state,done)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            agent.update()
            if done:
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'Explore: %.2f' % agent.epsilon)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
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
    plot(moving_average_rewards, ylabel="moving_average_rewards")
