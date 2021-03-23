#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2021-03-13 11:50:32
Discription: 
Environment: 
'''
import sys,os
sys.path.append(os.getcwd()) # 添加当前终端路径
from itertools import count
import datetime
import gym
from PolicyGradient.agent import PolicyGradient
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
    os.mkdir(RESULT_PATH)

class PGConfig:
    def __init__(self):
        self.train_eps = 300 # 训练的episode数目
        self.batch_size = 8
        self.lr = 0.01 # 学习率
        self.gamma = 0.99
        self.hidden_dim = 36 # 隐藏层维度
        
def train(cfg,env,agent):
    '''下面带pool都是存放的transition序列用于gradient'''
    state_pool = [] # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = [] 
    ''' 存储每个episode的reward用于绘图'''
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
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
                print('Episode:', i_episode, ' Reward:',  ep_reward)
                break
        if i_episode > 0 and i_episode % cfg.batch_size == 0:
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
            
if __name__ == "__main__":
    cfg = PGConfig()
    env = gym.make('CartPole-v0') # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent  = PolicyGradient(n_states,cfg)
    rewards, ma_rewards = train(cfg,env,agent)
    agent.save_model(SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = "Policy Gradient",path=RESULT_PATH)
