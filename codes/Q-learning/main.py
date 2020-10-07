#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2020-10-07 21:05:33
Discription: 
Environment: 
'''
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper
from agent import QLearning
import os
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
def get_args():
    '''训练的模型参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", default=0.9,
                        type=float, help="reward 的衰减率") 
    parser.add_argument("--epsilon_start", default=0.9,
                        type=float,help="e-greedy策略中初始epsilon")  
    parser.add_argument("--epsilon_end", default=0.1, type=float,help="e-greedy策略中的结束epsilon")
    parser.add_argument("--epsilon_decay", default=200, type=float,help="e-greedy策略中epsilon的衰减率")
    parser.add_argument("--policy_lr", default=0.1, type=float,help="学习率")
    parser.add_argument("--max_episodes", default=500, type=int,help="训练的最大episode数目") 

    config = parser.parse_args()

    return config

def train(cfg):
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    agent = QLearning(
        obs_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=cfg.policy_lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,epsilon_end=cfg.epsilon_end,epsilon_decay=cfg.epsilon_decay)
    render = False # 是否打开GUI画面
    rewards = [] # 记录所有episode的reward
    MA_rewards = []  # 记录滑动平均的reward
    steps = []# 记录所有episode的steps
    for i_episode in range(1,cfg.max_episodes+1):
        ep_reward = 0 # 记录每个episode的reward
        ep_steps = 0 # 记录每个episode走了多少step
        obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.sample(obs)  # 根据算法选择一个动作
            next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
            # 训练 Q-learning算法
            agent.learn(obs, action, reward, next_obs, done)  # 不需要下一步的action

            obs = next_obs  # 存储上一个观察值
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if render:
                env.render()  #渲染新的一帧图形
            if done:
                break
        steps.append(ep_steps)
        rewards.append(ep_reward)
        # 计算滑动平均的reward
        if i_episode == 1:
            MA_rewards.append(ep_reward)
        else:
            MA_rewards.append(
                0.9*MA_rewards[-1]+0.1*ep_reward) 
        print('Episode %s: steps = %s , reward = %.1f, explore = %.2f' % (i_episode, ep_steps,
                                                          ep_reward,agent.epsilon))                                 
        # 每隔20个episode渲染一下看看效果
        if i_episode % 20 == 0:
            render = True
        else:
            render = False
    agent.save() # 训练结束，保存模型

    output_path = os.path.dirname(__file__)+"/result/"
    # 检测是否存在文件夹
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(output_path+"rewards_train.npy", rewards)
    np.save(output_path+"MA_rewards_train.npy", MA_rewards)
    np.save(output_path+"steps_train.npy", steps)

def test(cfg):

    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    agent = QLearning(
        obs_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=cfg.policy_lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,epsilon_end=cfg.epsilon_end,epsilon_decay=cfg.epsilon_decay)
    agent.load() # 导入保存的模型
    rewards = [] # 记录所有episode的reward
    MA_rewards = []  # 记录滑动平均的reward
    steps = []# 记录所有episode的steps
    for i_episode in range(1,10+1):
        ep_reward = 0 # 记录每个episode的reward
        ep_steps = 0 # 记录每个episode走了多少step
        obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.predict(obs)  # 根据算法选择一个动作
            next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
            obs = next_obs  # 存储上一个观察值
            time.sleep(0.5)
            env.render()
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if done:
                break
        steps.append(ep_steps)
        rewards.append(ep_reward)
        # 计算滑动平均的reward
        if i_episode == 1:
            MA_rewards.append(ep_reward)
        else:
            MA_rewards.append(
                0.9*MA_rewards[-1]+0.1*ep_reward) 
        print('Episode %s: steps = %s , reward = %.1f' % (i_episode, ep_steps, ep_reward))
    plt.plot(MA_rewards)
    plt.show()   
def main():
    cfg = get_args()
    # train(cfg)
    test(cfg)

if __name__ == "__main__":
    main()