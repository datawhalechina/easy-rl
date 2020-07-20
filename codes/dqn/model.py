#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:47:02
@LastEditor: John
@LastEditTime: 2020-06-14 11:23:04
@Discription: 
@Environment: python 3.7.7
'''
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, n_states=4, n_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            n_states: number of features of input.
            n_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)