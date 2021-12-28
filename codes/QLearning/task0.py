#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-12-22 11:13:23
Discription: 
Environment: 
'''
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime

from envs.gridworld_env import CliffWalkingWapper
from QLearning.agent import QLearning
from QLearning.train import train,test
from common.utils import plot_rewards,plot_rewards_cn
from common.utils import save_results,make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
algo_name = 'Q-learning'  # 算法名称
env_name = 'CliffWalking-v0'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU
class QlearningConfig:
    '''训练相关参数'''
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.device = device # 检测GPU
        self.train_eps = 400 # 训练的回合数
        self.test_eps = 30 # 测试的回合数
        self.gamma = 0.9 # reward的衰减率
        self.epsilon_start = 0.95 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300 # e-greedy策略中epsilon的衰减率
        self.lr = 0.1 # 学习率      
class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        
def env_agent_config(cfg,seed=1):
    '''创建环境和智能体
    Args:
        cfg ([type]): [description]
        seed (int, optional): 随机种子. Defaults to 1.
    Returns:
        env [type]: 环境
        agent : 智能体
    '''    
    env = gym.make(cfg.env_name)  
    env = CliffWalkingWapper(env)
    env.seed(seed) # 设置随机种子
    state_dim = env.observation_space.n # 状态维度
    action_dim = env.action_space.n # 动作维度
    agent = QLearning(state_dim,action_dim,cfg)
    return env,agent

cfg = QlearningConfig()
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
        
    
