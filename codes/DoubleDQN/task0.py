#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-11-07 18:10:37
LastEditor: JiangJi
LastEditTime: 2021-11-19 18:34:05
Discription: 
'''

import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import datetime

from common.utils import save_results, make_dir
from common.utils import plot_rewards
from DoubleDQN.agent import DoubleDQN
from DoubleDQN.train import train,test

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'DoubleDQN'  # 算法名称
env_name = 'CartPole-v0'  # 环境名称
class DoubleDQNConfig:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
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
    agent = DoubleDQN(state_dim,action_dim,cfg)
    return env,agent

cfg = DoubleDQNConfig()
plot_cfg = PlotConfig()
# 训练
env,agent = env_agent_config(cfg,seed=1)
rewards, ma_rewards = train(cfg, env, agent)
make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
agent.save(path=plot_cfg.model_path)  # 保存模型
save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
# 测试
env,agent = env_agent_config(cfg,seed=10)
agent.load(path=plot_cfg.model_path)  # 导入模型
rewards,ma_rewards = test(cfg,env,agent)
save_results(rewards, ma_rewards, tag='test', path=plot_cfg.result_path)  # 保存结果
plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果
