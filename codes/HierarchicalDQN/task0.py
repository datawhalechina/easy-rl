#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-29 10:37:32
LastEditor: John
LastEditTime: 2021-05-04 22:35:56
Discription: 
Environment: 
'''
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import datetime
import numpy as np
import torch
import gym

from common.utils import save_results,make_dir
from common.utils import plot_rewards
from HierarchicalDQN.agent import HierarchicalDQN
from HierarchicalDQN.train import train,test

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = "Hierarchical DQN"  # 算法名称
env_name = 'CartPole-v0'  # 环境名称
class HierarchicalDQNConfig:
    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 300  # 训练的episode数目
        self.test_eps = 50  # 测试的episode数目
        self.gamma = 0.99
        self.epsilon_start = 1  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 10000  # Replay Memory capacity
        self.batch_size = 32
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层
class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片

def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env_name)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HierarchicalDQN(state_dim,action_dim,cfg)
    return env,agent

if __name__ == "__main__":
    cfg = HierarchicalDQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                path=plot_cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=plot_cfg.model_path)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test', path=plot_cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

