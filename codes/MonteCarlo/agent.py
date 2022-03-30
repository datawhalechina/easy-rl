#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:14:34
LastEditor: John
LastEditTime: 2021-05-05 16:58:39
Discription: 
Environment: 
'''
import numpy as np
from collections import defaultdict
import torch
import dill

class FisrtVisitMC:
    ''' On-Policy First-Visit MC Control
    '''
    def __init__(self,action_dim,cfg):
        self.action_dim = action_dim
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma 
        self.Q_table = defaultdict(lambda: np.zeros(action_dim))
        self.returns_sum = defaultdict(float) # sum of returns
        self.returns_count = defaultdict(float)
        
    def choose_action(self,state):
        ''' e-greed policy '''
        if state in self.Q_table.keys():
            best_action = np.argmax(self.Q_table[state])
            action_probs = np.ones(self.action_dim, dtype=float) * self.epsilon / self.action_dim
            action_probs[best_action] += (1.0 - self.epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        else:
            action = np.random.randint(0,self.action_dim)
        return action
    def update(self,one_ep_transition):
        # Find all (state, action) pairs we've visited in this one_ep_transition
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in one_ep_transition])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurence of the (state, action) pair in the one_ep_transition
            first_occurence_idx = next(i for i,x in enumerate(one_ep_transition)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(self.gamma**i) for i,x in enumerate(one_ep_transition[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            self.Q_table[state][action] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]
    def save(self,path):
        '''把 Q表格 的数据保存到文件中
        '''
        torch.save(
            obj=self.Q_table,
            f=path+"Q_table",
            pickle_module=dill
        )

    def load(self, path):
        '''从文件中读取数据到 Q表格
        '''
        self.Q_table =torch.load(f=path+"Q_table",pickle_module=dill)