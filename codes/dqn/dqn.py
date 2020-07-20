#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
@LastEditTime: 2020-06-14 13:56:45
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from memory import ReplayBuffer
from model import FCN

class DQN:
    def __init__(self, n_states, n_actions, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, memory_capacity=10000, policy_lr=0.01,batch_size=128, device="cpu"):
        self.actions_count = 0
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.policy_net = FCN(n_states,n_actions).to(self.device)
        self.target_net = FCN(n_states,n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=policy_lr)
        self.loss = 0
        self.memory = ReplayBuffer(memory_capacity)

    def select_action(self,state):
        '''选择工作
        Args:
            state [array]: 状态
        Returns:
            [array]: 动作
        '''
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.actions_count / self.epsilon_decay)
        self.actions_count += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                state   = torch.tensor([state],device=self.device,dtype=torch.float32) # 先转为张量便于丢给神经网络,state元素数据原本为float64；注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                q_value = self.policy_net(state) # tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action
    def update(self):

        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(state_batch,device=self.device,dtype=torch.float) # 例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])
        action_batch  = torch.tensor(action_batch,device=self.device).unsqueeze(1) # 例如tensor([[1],...,[0]])
        reward_batch  = torch.tensor(reward_batch,device=self.device,dtype=torch.float) # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(next_state_batch,device=self.device,dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch),device=self.device).unsqueeze(1) # 将bool转为float然后转为张量
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values = self.policy_net(state_batch).gather(1, action_batch) # 等价于self.forward       
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
     
        next_state_values = self.target_net(
            next_state_batch).max(1)[0].detach() # tensor([ 0.0060, -0.0171,...,])
        # Compute the expected Q values
        expected_q_values = reward_batch + self.gamma * next_state_values * (1-done_batch[0])

        # Compute Huber loss
        # self.loss = nn.MSELoss(q_values, expected_q_values.unsqueeze(1)) 
        self.loss = nn.MSELoss()(q_values,expected_q_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad() # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        self.loss.backward() # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        for param in self.policy_net.parameters(): # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
        

