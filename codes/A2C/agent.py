#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-03 20:47:09
LastEditor: John
LastEditTime: 2021-03-20 17:41:21
Discription: 
Environment: 
'''
from A2C.model import ActorCritic
import torch.optim as optim

class A2C:
    def __init__(self,state_dim, action_dim, cfg):
        self.gamma = 0.99
        self.model = ActorCritic(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=cfg.lr)
    def choose_action(self, state):
        dist, value = self.model(state)
        action = dist.sample()
        return action
    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    def update(self):
        pass