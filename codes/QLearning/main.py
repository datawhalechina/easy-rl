#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-03-11 19:22:50
Discription: 
Environment: 
'''

import sys,os
sys.path.append(os.getcwd()) # 添加当前终端路径
import argparse
import gym
import datetime
from QLearning.plot import plot
from QLearning.utils import save_results
from envs.gridworld_env import CliffWalkingWapper, FrozenLakeWapper
from QLearning.agent import QLearning

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''训练的模型参数
    '''
    parser = argparse.ArgumentParser()
    '''训练相关参数'''
    parser.add_argument("--n_episodes", default=500,
                        type=int, help="训练的最大episode数目")       
    '''算法相关参数'''
    parser.add_argument("--gamma", default=0.9,
                        type=float, help="reward的衰减率")
    parser.add_argument("--epsilon_start", default=0.99,
                        type=float, help="e-greedy策略中初始epsilon")
    parser.add_argument("--epsilon_end", default=0.01,
                        type=float, help="e-greedy策略中的结束epsilon")
    parser.add_argument("--epsilon_decay", default=200,
                        type=float, help="e-greedy策略中epsilon的衰减率")
    parser.add_argument("--lr", default=0.1, type=float, help="学习率")
    config = parser.parse_args()
    return config
def train(cfg,env,agent):
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    rewards = []  # 记录所有episode的reward,
    steps = []  # 记录所有episode的steps
    for i_episode in range(cfg.n_episodes):
        ep_reward = 0  # 记录每个episode的reward
        ep_steps = 0  # 记录每个episode走了多少step
        obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.choose_action(obs)  # 根据算法选择一个动作
            next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
            # 训练 Q-learning算法
            agent.update(obs, action, reward, next_obs, done)  # 不需要下一步的action
            obs = next_obs  # 存储上一个观察值
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if done:
                break
        steps.append(ep_steps)
        # 计算滑动平均的reward
        if rewards:
            rewards.append(rewards[-1]*0.9+ep_reward*0.1)
        else:
            rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_episode+1, cfg.n_episodes,ep_reward))
    plot(rewards)
    if not os.path.exists(SAVED_MODEL_PATH):
        os.mkdir(SAVED_MODEL_PATH)
    agent.save(SAVED_MODEL_PATH+'Q_table.pkl')  # 训练结束，保存模型
    '''存储reward等相关结果'''
    save_results(rewards,tag='train',result_path=RESULT_PATH)

def eval(cfg,env,agent):
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)
    rewards = []  # 记录所有episode的reward,
    steps = []  # 记录所有episode的steps
    for i_episode in range(20):
        ep_reward = 0  # 记录每个episode的reward
        ep_steps = 0  # 记录每个episode走了多少step
        obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.choose_action(obs)  # 根据算法选择一个动作
            next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
            obs = next_obs  # 存储上一个观察值
            ep_reward += reward
            ep_steps += 1  # 计算step数
            if done:
                break
        steps.append(ep_steps)
        # 计算滑动平均的reward
        if rewards:
            rewards.append(rewards[-1]*0.9+ep_reward*0.1)
        else:
            rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_episode+1, cfg.n_episodes,ep_reward))
    plot(rewards)
    '''存储reward等相关结果'''
    save_results(rewards,tag='eval',result_path=RESULT_PATH)
    
if __name__ == "__main__":
    cfg = get_args()
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    n_actions = env.action_space.n
    agent = QLearning(n_actions,cfg)
    train(cfg,env,agent)
    eval(cfg,env,agent)
    
