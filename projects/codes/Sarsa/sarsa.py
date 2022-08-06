#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:58:16
LastEditor: John
LastEditTime: 2022-04-29 20:12:57
Discription: 
Environment: 
'''
import numpy as np
from collections import defaultdict
import torch
import math
class Sarsa(object):
    def __init__(self,
                 n_actions,cfg,):
        self.n_actions = n_actions  
        self.lr = cfg.lr  
        self.gamma = cfg.gamma  
        self.sample_count = 0 
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay 
        self.Q  = defaultdict(lambda: np.zeros(n_actions)) # Q table
    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # The probability to select a random action, is is log decayed
        best_action = np.argmax(self.Q[state])
        action_probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) 
        return action
    def predict_action(self,state):
        return np.argmax(self.Q[state])
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