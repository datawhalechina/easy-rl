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


import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import datetime
import numpy as np
import torch
import gym

from common.utils import save_results,make_dir
from common.plot import plot_rewards
from HierarchicalDQN.agent import HierarchicalDQN

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class HierarchicalDQNConfig:
    def __init__(self):
        self.algo = "H-DQN"  # name of algo
        self.env = 'CartPole-v0'
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # path to save models
        self.train_eps = 300  # 训练的episode数目
        self.eval_eps = 50  # 测试的episode数目
        self.gamma = 0.99
        self.epsilon_start = 1  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 10000  # Replay Memory capacity
        self.batch_size = 32
        self.target_update = 2  # target net的更新频率
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测gpu
        self.hidden_dim = 256  # dimension of hidden layer

def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HierarchicalDQN(state_dim,action_dim,cfg)
    return env,agent

def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            goal = agent.set_goal(state)
            onehot_goal = agent.to_onehot(goal)
            meta_state = state
            extrinsic_reward = 0
            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onehot_goal])
                action = agent.choose_action(goal_state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == np.argmax(
                    next_state) else 0.0
                agent.memory.push(goal_state, action, intrinsic_reward, np.concatenate(
                    [next_state, onehot_goal]), done)
                state = next_state
                agent.update()
        agent.meta_memory.push(meta_state, goal, extrinsic_reward, state, done)
        print('Episode:{}/{}, Reward:{}, Loss:{:.2f}, Meta_Loss:{:.2f}'.format(i_ep+1, cfg.train_eps, ep_reward,agent.loss_numpy ,agent.meta_loss_numpy ))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            goal = agent.set_goal(state)
            onehot_goal = agent.to_onehot(goal)
            extrinsic_reward = 0
            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onehot_goal])
                action = agent.choose_action(goal_state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                extrinsic_reward += reward
                state = next_state
                agent.update()
        print(f'Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward}, Loss:{agent.loss_numpy:.2f}, Meta_Loss:{agent.meta_loss_numpy:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg = HierarchicalDQNConfig()

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

