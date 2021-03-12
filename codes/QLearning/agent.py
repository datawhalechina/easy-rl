#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-03-12 16:48:25
Discription: 
Environment: 
'''
import numpy as np
import math
import torch
from collections import defaultdict

class QLearning(object):
    def __init__(self,
                 n_actions,cfg):
        self.n_actions = n_actions  # number of actions
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  
        self.epsilon = 0 
        self.sample_count = 0  # epsilon随训练的也就是采样次数逐渐衰减，所以需要计数
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(n_actions)) # 使用字典存储Q表，个人比较喜欢这种，也可以用下面一行的二维数组表示，但是需要额外更改代码
        # self.Q_table = np.zeros((n_states, n_actions))  # Q表
    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        # 随机选取0-1之间的值，如果大于epsilon就按照贪心策略选取action，否则随机选取
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[state])
        else:
            action = np.random.choice(self.n_actions)  # 有一定概率随机探索选取一个动作
        return action
            
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * np.max(
                self.Q_table[next_state])  # Q_table-learning
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)
    def save(self,path):
        '''把 Q表格 的数据保存到文件中
        '''
        import dill
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)