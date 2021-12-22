#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2021-10-16 00:34:13
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path

import gym
import torch
import datetime
from itertools import count

from PolicyGradient.agent import PolicyGradient
from common.plot import plot_rewards
from common.utils import save_results,make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class PGConfig:
    def __init__(self):
        self.algo = "PolicyGradient"  # 算法名称
        self.env = 'CartPole-v0' # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.train_eps = 300 # 训练的回合数
        self.test_eps = 30 # 测试的回合数
        self.batch_size = 8
        self.lr = 0.01 # 学习率
        self.gamma = 0.99
        self.hidden_dim = 36 # dimmension of hidden layer
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu


def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    agent = PolicyGradient(state_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    state_pool = [] # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = [] 
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', i_ep, ' Reward:',  ep_reward)
                break
        if i_ep > 0 and i_ep % cfg.batch_size == 0:
            agent.update(reward_pool,state_pool,action_pool)
            state_pool = [] # 每个episode的state
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('complete training！')
    return rewards, ma_rewards
            

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state = next_state
            if done:
                print('Episode:', i_ep, ' Reward:',  ep_reward)
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('complete evaling！')
    return rewards, ma_rewards
    
if __name__ == "__main__":
    cfg = PGConfig()

     # train
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    # eval
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)

