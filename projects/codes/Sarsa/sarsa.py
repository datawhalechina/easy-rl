#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:58:16
LastEditor: John
LastEditTime: 2022-10-30 02:00:51
Discription: 
Environment: 
'''
import numpy as np
from collections import defaultdict
import torch
import math
class Sarsa(object):
    def __init__(self,cfg):
        self.n_actions = cfg.n_actions 
        self.lr = cfg.lr 
        self.gamma = cfg.gamma    
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(self.n_actions)) # Q table
    def sample_action(self, state):
        ''' another way to represent e-greedy policy
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # The probability to select a random action, is is log decayed
        best_action = np.argmax(self.Q_table[str(state)]) # array cannot be hashtable, thus convert to str
        action_probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        action_probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) 
        return action
    def predict_action(self,state):
        ''' predict action while testing 
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, next_action,done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action] # the only difference from Q learning
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict) 
    def save_model(self,path):
        import dill
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=path+"checkpoint.pkl",
            pickle_module=dill
        )
        print("Model saved!")
    def load_model(self, path):
        import dill
        self.Q_table=torch.load(f=path+'checkpoint.pkl',pickle_module=dill)
        print("Mode loaded!")