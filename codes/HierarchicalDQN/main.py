#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-29 10:37:32
LastEditor: John
LastEditTime: 2021-03-31 14:58:49
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

from common.utils import save_results
from common.plot import plot_rewards,plot_losses
from HierarchicalDQN.agent import HierarchicalDQN

SEQUENCE = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time
SAVED_MODEL_PATH = curr_path+"/saved_model/"+SEQUENCE+'/'  # path to save model
if not os.path.exists(curr_path+"/saved_model/"):
    os.mkdir(curr_path+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = curr_path+"/results/"+SEQUENCE+'/'  # path to save rewards
if not os.path.exists(curr_path+"/results/"):
    os.mkdir(curr_path+"/results/")
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


class HierarchicalDQNConfig:
    def __init__(self):
        self.algo = "H-DQN"  # name of algo
        self.gamma = 0.99
        self.epsilon_start = 1  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 10000  # Replay Memory capacity
        self.batch_size = 32
        self.train_eps = 300  # 训练的episode数目
        self.target_update = 2  # target net的更新频率
        self.eval_eps = 20  # 测试的episode数目
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测gpu
        self.hidden_dim = 256  # dimension of hidden layer


def train(cfg, env, agent):
    print('Start to train !')
    rewards = []
    ma_rewards = []  # moveing average reward
    for i_episode in range(cfg.train_eps):
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
        print('Episode:{}/{}, Reward:{}, Loss:{:.2f}, Meta_Loss:{:.2f}'.format(i_episode+1, cfg.train_eps, ep_reward,agent.loss_numpy ,agent.meta_loss_numpy ))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.seed(1)
    cfg = HierarchicalDQNConfig()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = HierarchicalDQN(state_dim, action_dim, cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards, ma_rewards, tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=RESULT_PATH)
    plot_losses(agent.losses,algo=cfg.algo, path=RESULT_PATH)

