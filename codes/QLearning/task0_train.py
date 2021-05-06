#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-05-06 17:04:38
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import gym
import torch
import datetime

from envs.gridworld_env import CliffWalkingWapper
from QLearning.agent import QLearning
from common.plot import plot_rewards
from common.utils import save_results,make_dir
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class QlearningConfig:
    '''训练相关参数'''
    def __init__(self):
        self.algo = 'Qlearning'
        self.env = 'CliffWalking-v0' # 0 up, 1 right, 2 down, 3 left
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 300 # 训练的episode数目
        self.eval_eps = 30
        self.gamma = 0.9 # reward的衰减率
        self.epsilon_start = 0.95 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 200 # e-greedy策略中epsilon的衰减率
        self.lr = 0.1 # learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu

        
def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env = CliffWalkingWapper(env)
    env.seed(seed)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = QLearning(state_dim,action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []  
    ma_rewards = [] # moving average reward
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
    print('Complete training！')
    return rewards,ma_rewards
    
def eval(cfg,env,agent):
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []  # 记录所有episode的reward
    ma_rewards = [] # 滑动平均的reward
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"Episode:{i_ep+1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return rewards,ma_rewards
    
if __name__ == "__main__":
    cfg = QlearningConfig()
    env,agent = env_agent_config(cfg,seed=1)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(cfg.result_path,cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)

    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
    
    
