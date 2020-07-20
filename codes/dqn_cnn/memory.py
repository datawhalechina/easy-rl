#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 09:42:44
@LastEditor: John
@LastEditTime: 2020-06-11 15:50:33
@Discription: 
@Environment: python 3.7.7
'''
from collections import namedtuple
import random


                      
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
