#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-09-16 01:31:33
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径sys.path

import datetime
import gym
import torch

from DDPG.env import NormalizedActions
from DDPG.agent import DDPG
from DDPG.train import train,test
from common.utils import save_results,make_dir
from common.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'DDPG'  # 算法名称
env_name = 'Pendulum-v1'  # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1

class DDPGConfig:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU
        self.train_eps = 300 # 训练的回合数
        self.test_eps = 50 # 测试的回合数
        self.gamma = 0.99 # 折扣因子
        self.critic_lr = 1e-3 # 评论家网络的学习率
        self.actor_lr = 1e-4 # 演员网络的学习率
        self.memory_capacity = 8000 # 经验回放的容量
        self.batch_size = 128 # mini-batch SGD中的批量大小
        self.target_update = 2 # 目标网络的更新频率
        self.hidden_dim = 256 # 网络隐藏层维度
        self.soft_tau = 1e-2 # 软更新参数
        
class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make(cfg.env_name)) # 装饰action噪声
    env.seed(seed) # 随机种子
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,cfg)
    return env,agent

cfg = DDPGConfig()
plot_cfg = PlotConfig()
# 训练
env,agent = env_agent_config(cfg,seed=1)
rewards, ma_rewards = train(cfg, env, agent)
make_dir(plot_cfg.result_path, plot_cfg.model_path)
agent.save(path=plot_cfg.model_path)
save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
# 测试
env,agent = env_agent_config(cfg,seed=10)
agent.load(path=plot_cfg.model_path)
rewards,ma_rewards = test(plot_cfg,env,agent)
save_results(rewards,ma_rewards,tag = 'test',path = cfg.result_path)
plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

