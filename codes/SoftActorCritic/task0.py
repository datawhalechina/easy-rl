#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-29 12:59:22
LastEditor: JiangJi
LastEditTime: 2021-12-22 16:27:13
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import datetime

from SoftActorCritic.env_wrapper import NormalizedActions
from SoftActorCritic.sac import SAC
from common.utils import save_results, make_dir
from common.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'SAC'  # 算法名称
env_name = 'Pendulum-v1'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

class SACConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name
        self.env_name = env_name # 环境名称
        self.device= device
        self.train_eps = 300
        self.test_eps = 20
        self.max_steps = 500 # 每回合的最大步数
        self.gamma = 0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2
        self.value_lr  = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size  = 128
        

class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device= device
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make(cfg.env_name))
    env.seed(seed)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    agent = SAC(state_dim,action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0 # 记录一回合内的奖励
        state = env.reset() # 重置环境，返回初始状态
        for i_step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
        if (i_ep+1)%10 == 0: 
            print(f'回合：{i_ep+1}/{cfg.train_eps}, 奖励：{ep_reward:.3f}')
    print('完成训练！')
    return rewards, ma_rewards

def test(cfg,env,agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        for i_step in range(cfg.max_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg=SACConfig()
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




