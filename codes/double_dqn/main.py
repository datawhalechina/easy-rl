#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2020-09-01 22:54:23
@Discription: 
@Environment: python 3.7.7
'''
import gym
import torch
from dqn import DQN
from plot import plot
import argparse

def get_args():
    '''模型参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.99,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.95,
                        type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.05, type=float)
    parser.add_argument("--epsilon_decay", default=500, type=float)
    parser.add_argument("--policy_lr", default=0.01, type=float)
    parser.add_argument("--memory_capacity", default=1000,
                        type=int, help="capacity of Replay Memory") 

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int) # 训练的最大episode数目
    parser.add_argument("--train_steps", default=200, type=int) # 训练每个episode的长度
    parser.add_argument("--eval_eps", default=200, type=int) # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200, type=int) # 训练每个episode的长度
    parser.add_argument("--target_update", default=2, type=int,
                        help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()

    return config

def train():
    cfg = get_args()
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    for i_episode in range(1, cfg.train_eps+1):
        state = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.train_steps+1):
            action = agent.select_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
            state = next_state # 跳转到下一个状态
            agent.update() # 每步更新网络
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
    import os
    import numpy as np
    save_path = os.path.dirname(__file__)+"/saved_model/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    agent.save_model(save_path+'checkpoint.pth')
    # 存储reward等相关结果
    output_path = os.path.dirname(__file__)+"/result/"
    # 检测是否存在文件夹
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(output_path+"rewards.npy", rewards)
    np.save(output_path+"moving_average_rewards.npy", moving_average_rewards)
    np.save(output_path+"steps.npy", ep_steps)
    print('Complete！')
    plot(rewards)
    plot(moving_average_rewards, ylabel="moving_average_rewards")
    plot(ep_steps, ylabel="steps_of_each_episode")

def eval():
    cfg = get_args()
    # if gpu is to be used
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
    import os
    save_path = os.path.dirname(__file__)+"/saved_model/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    agent.load_model(save_path+'checkpoint.pth')
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    for i_episode in range(1, cfg.eval_eps+1):
        state = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.eval_steps+1):
            action = agent.select_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            state = next_state # 跳转到下一个状态
            if done:
                break
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
    plot(rewards,save_fig=False)
    plot(moving_average_rewards, ylabel="moving_average_rewards",save_fig=False)
    plot(ep_steps, ylabel="steps_of_each_episode",save_fig=False)
if __name__ == "__main__":
    # train()
    eval()
