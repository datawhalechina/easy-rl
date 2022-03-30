#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:58:16
LastEditor: John
LastEditTime: 2021-03-13 11:02:50
Discription: 
Environment: 
'''
import numpy as np
from collections import defaultdict
import torch
class Sarsa(object):
    def __init__(self,
                 action_dim,sarsa_cfg,):
        self.action_dim = action_dim  # number of actions
        self.lr = sarsa_cfg.lr  # learning rate
        self.gamma = sarsa_cfg.gamma  
        self.epsilon = sarsa_cfg.epsilon  
        self.Q  = defaultdict(lambda: np.zeros(action_dim))
        # self.Q = np.zeros((state_dim, action_dim))  # Q表
    def choose_action(self, state):
        best_action = np.argmax(self.Q[state])
        # action = best_action
        action_probs = np.ones(self.action_dim, dtype=float) * self.epsilon / self.action_dim
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) 
        return action
            
    def update(self, state, action, reward, next_state, next_action,done):
        Q_predict = self.Q[state][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * self.Q[next_state][next_action] 
        self.Q[state][action] += self.lr * (Q_target - Q_predict) 
    def save(self,path):
        '''把 Q表格 的数据保存到文件中
        '''
        import dill
        torch.save(
            obj=self.Q,
            f=path+"sarsa_model.pkl",
            pickle_module=dill
        )
    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        import dill
        self.Q =torch.load(f=path+'sarsa_model.pkl',pickle_module=dill)