#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-03-31 18:14:59
Discription: 
Environment: 
'''

import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import gym
import datetime

from envs.gridworld_env import CliffWalkingWapper, FrozenLakeWapper
from QLearning.agent import QLearning
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = curr_path+"/saved_model/"+SEQUENCE+'/' # path to save model
if not os.path.exists(curr_path+"/saved_model/"): 
    os.mkdir(curr_path+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): 
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = curr_path+"/results/"+SEQUENCE+'/' # path to save rewards
if not os.path.exists(curr_path+"/results/"): 
    os.mkdir(curr_path+"/results/")
if not os.path.exists(RESULT_PATH): 
    os.mkdir(RESULT_PATH)

class QlearningConfig:
    '''训练相关参数'''
    def __init__(self):
        self.train_eps = 200 # 训练的episode数目
        self.gamma = 0.9 # reward的衰减率
        self.epsilon_start = 0.99 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 200 # e-greedy策略中epsilon的衰减率
        self.lr = 0.1 # learning rate

def train(cfg,env,agent):
    rewards = []  
    ma_rewards = [] # moving average reward
    steps = []  # 记录所有episode的steps
    for i_episode in range(cfg.train_eps):
        ep_reward = 0  # 记录每个episode的reward
        ep_steps = 0  # 记录每个episode走了多少step
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if done:
                break
        steps.append(ep_steps)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_episode+1, cfg.train_eps,ep_reward))
    return rewards,ma_rewards

def eval(cfg,env,agent):
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    rewards = []  # 记录所有episode的reward
    ma_rewards = [] # 滑动平均的reward
    steps = []  # 记录所有episode的steps
    for i_episode in range(cfg.train_eps):
        ep_reward = 0  # 记录每个episode的reward
        ep_steps = 0  # 记录每个episode走了多少step
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if done:
                break
        steps.append(ep_steps)
        rewards.append(ep_reward)
        # 计算滑动平均的reward
        if ma_rewards:
            ma_rewards.append(rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_episode+1, cfg.train_eps,ep_reward))
    return rewards,ma_rewards
    
if __name__ == "__main__":
    cfg = QlearningConfig()
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    action_dim = env.action_space.n
    agent = QLearning(action_dim,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = "On-Policy First-Visit MC Control",path=RESULT_PATH)
    
    
