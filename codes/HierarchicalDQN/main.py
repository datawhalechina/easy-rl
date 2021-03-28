#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-24 22:14:04
LastEditor: John
LastEditTime: 2021-03-27 04:23:43
Discription: 
Environment: 
'''
import sys,os
sys.path.append(os.getcwd()) # add current terminal path to sys.path
import gym
import numpy as np
import torch
import datetime
from HierarchicalDQN.agent import HierarchicalDQN
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'  # path to save model
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # path to save rewards
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): 
    os.mkdir(RESULT_PATH)

class HierarchicalDQNConfig:
    def __init__(self):
        self.algo = "DQN" # name of algo
        self.gamma = 0.99
        self.epsilon_start = 0.95 # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.01 # learning rate
        self.memory_capacity = 800 # Replay Memory capacity
        self.batch_size = 64
        self.train_eps = 250 # 训练的episode数目
        self.train_steps = 200 # 训练每个episode的最大长度
        self.target_update = 2 # target net的更新频率
        self.eval_eps = 20 # 测试的episode数目
        self.eval_steps = 200 # 测试每个episode的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
        self.hidden_dim = 256 # dimension of hidden layer

def train(cfg,env,agent):
    print('Start to train !')
    rewards = []
    ma_rewards = [] # moving average reward
    ep_steps = []
    for i_episode in range(cfg.train_eps):
        state = env.reset() 
        extrinsic_reward = 0
        for i_step in range(cfg.train_steps):
            goal= agent.set_goal(state)
            meta_state = state
            goal_state  = np.concatenate([state, goal])
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)
            extrinsic_reward += reward
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0
            agent.memory.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, goal]), done)
            state = next_state 
            agent.update()
            if done:
                break
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:{}/{}, Reward:{}, Steps:{}, Done:{}'.format(i_episode+1,cfg.train_eps,extrinsic_reward,i_step+1,done))
        ep_steps.append(i_step)
        rewards.append(extrinsic_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*extrinsic_reward)
        else:
            ma_rewards.append(extrinsic_reward)   
    agent.meta_memory.push(meta_state, goal, extrinsic_reward, state, done)
    print('Complete training！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = HierarchicalDQNConfig()
    env = gym.make('CartPole-v0')
    env.seed(1) 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HierarchicalDQN(state_dim,action_dim,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)