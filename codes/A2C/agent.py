#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-05-03 22:16:08
LastEditor: JiangJi
LastEditTime: 2021-05-03 22:23:48
Discription: 
Environment: 
'''
import torch.optim as optim
from A2C.model import ActorCritic
class A2C:
    def __init__(self,n_states,n_actions,cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(n_states, n_actions, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns