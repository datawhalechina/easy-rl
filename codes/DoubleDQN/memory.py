#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:27:16
@LastEditor: John
LastEditTime: 2020-12-22 12:56:27
@Discription: 
@Environment: python 3.7.7
'''
import random
import numpy as np

class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity # buffer的最大容量
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        '''以队列的方式将样本填入buffer中
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        '''随机采样batch_size个样本
        '''
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =  zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        '''返回buffer的长度
        '''
        return len(self.buffer)

