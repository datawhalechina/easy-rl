#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-03 20:47:09
LastEditor: John
LastEditTime: 2020-11-08 22:16:29
Discription: 
Environment: 
'''
from model import ActorCritic
import torch.optim as optim

class A2C:
    def __init__(self,n_states, n_actions, hidden_dim=256,device="cpu",lr = 3e-4):
        self.device = device
        self.gamma = 0.99
        self.model = ActorCritic(n_states, n_actions, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
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