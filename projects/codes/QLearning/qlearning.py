#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-08-24 10:31:04
Discription: use defaultdict to define Q table
Environment: 
'''
import numpy as np
import math
import torch
from collections import defaultdict

class QLearning(object):
    def __init__(self,cfg):
        self.n_actions = cfg['n_actions'] 
        self.lr = cfg['lr']  
        self.gamma = cfg['gamma']  
        self.epsilon = cfg['epsilon_start']
        self.sample_count = 0  
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.Q_table  = defaultdict(lambda: np.zeros(self.n_actions)) # use nested dictionary to represent Q(s,a), here set all Q(s,a)=0 initially, not like pseudo code
    def sample_action(self, state):
        ''' sample action with e-greedy policy while training
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # choose action corresponding to the maximum q value
        else:
            action = np.random.choice(self.n_actions) # choose action randomly
        return action
    def predict_action(self,state):
        ''' predict action while testing 
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action] 
        if done: # terminal state
            Q_target = reward  
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    def save_model(self,path):
        import dill
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
        print("Model saved!")
    def load_model(self, path):
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)
        print("Mode loaded!")