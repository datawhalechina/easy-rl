#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-12-22 11:14:17
LastEditor: JiangJi
LastEditTime: 2021-12-22 11:40:44
Discription: 使用 Nature DQN 训练 CartPole-v1
'''
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime
from common.utils import save_results, make_dir
from common.utils import plot_rewards, plot_rewards_cn
from DQN.dqn import DQN

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = "DQN"  # 算法名称
env_name = 'CartPole-v1'  # 环境名称
class DQNConfig:
    ''' 算法相关参数设置
    '''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        # 超参数
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
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


def env_agent_config(cfg, seed=1):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    env.seed(seed)  # 设置随机种子
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n  # 动作维度
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    return env, agent

def train(cfg, env, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0 # 记录一回合内的奖励
        state = env.reset() # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state) # 选择动作
            next_state, reward, done, _ = env.step(action) # 更新环境，返回transition
            agent.memory.push(state, action, reward, next_state, done) # 保存transition
            state = next_state # 更新下一个状态
            agent.update() # 更新智能体
            ep_reward += reward # 累加奖励
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0: # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%10 == 0: 
            print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards

def test(cfg,env,agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    # 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0
    cfg.epsilon_start = 0.0 # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0 # e-greedy策略中的终止epsilon
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0 # 记录一回合内的奖励
        state = env.reset() # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state) # 选择动作
            next_state, reward, done, _ = env.step(action) # 更新环境，返回transition
            state = next_state # 更新下一个状态
            ep_reward += reward # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards,ma_rewards
if __name__ == "__main__":
    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                path=plot_cfg.result_path)  # 保存结果
    plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=plot_cfg.model_path)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test',
                path=plot_cfg.result_path)  # 保存结果
    plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果
