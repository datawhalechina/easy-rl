#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-01-05 09:41:34
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
from env import CliffWalkingWapper, FrozenLakeWapper
from agent import QLearning
import os
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from env import env_init_1
from params import get_args
from params import SEQUENCE, SAVED_MODEL_PATH, RESULT_PATH
from utils import save_results,save_model
from plot import plot

def train(cfg):
    '''# env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)'''
    env = env_init_1()
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
        '''计算滑动平均的reward'''
        if i_episode == 1:
            MA_rewards.append(ep_reward)
        else:
            MA_rewards.append(
                0.9*MA_rewards[-1]+0.1*ep_reward) 
        print('Episode %s: steps = %s , reward = %.1f, explore = %.2f' % (i_episode, ep_steps,
                                                          ep_reward,agent.epsilon))                                 
        '''每隔20个episode渲染一下看看效果'''
        if i_episode % 20 == 0:
            render = True
        else:
            render = False
    print('Complete training！')
    save_model(agent,model_path=SAVED_MODEL_PATH)
    '''存储reward等相关结果'''
    save_results(rewards,MA_rewards,tag='train',result_path=RESULT_PATH)
    plot(rewards)
    plot(MA_rewards,ylabel='moving_average_rewards_train')

def eval(cfg, saved_model_path = SAVED_MODEL_PATH):

    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    agent = QLearning(
        obs_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=cfg.policy_lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,epsilon_end=cfg.epsilon_end,epsilon_decay=cfg.epsilon_decay)
    agent.load_model(saved_model_path+'checkpoint.npy') # 导入保存的模型
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
    print('Complete training！')
    save_model(agent,model_path=SAVED_MODEL_PATH)
    '''存储reward等相关结果'''
    save_results(rewards,MA_rewards,tag='train',result_path=RESULT_PATH)
    plot(rewards)
    plot(MA_rewards,ylabel='moving_average_rewards_train')
      
if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)