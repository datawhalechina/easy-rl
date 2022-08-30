#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-08-29 23:34:20
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
class DoubleDQN:
    def __init__(self,models, memories, cfg):
        self.n_actions = cfg['n_actions']  
        self.device = torch.device(cfg['device']) 
        self.gamma = cfg['gamma'] 
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.policy_net = models['Qnet'].to(self.device)
        self.target_net = models['Qnet'].to(self.device)
        # target_net copy from policy_net
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # self.target_net.eval()  # donnot use BatchNormalization or Dropout
        # the difference between parameters() and state_dict() is that parameters() require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg['lr'])
        self.memory = memories['Memory']
        self.update_flag = False 
        
    def sample_action(self, state):
        ''' sample action
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.sample_count / self.epsilon_decay)
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_value = self.policy_net(state)
                action = q_value.max(1)[1].item()  
        else:
            action = random.randrange(self.n_actions)
        return action
    def predict_action(self, state):
        ''' predict action
        '''
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            q_value = self.policy_net(state)
            action = q_value.max(1)[1].item()  
        return action
    def update(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        # convert to tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1)  # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        # compute current Q(s_t|a=a_t)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_q_value_batch = self.policy_net(next_state_batch)
        '''the following is the way of computing Double DQN expected_q_value，a bit different from Nature DQN'''
        next_target_value_batch = self.target_net(next_state_batch)
        # choose action a from Q(s_t‘, a), next_target_values obtain next_q_value，which is Q’(s_t|a=argmax Q(s_t‘, a))
        next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(1)) # shape(batchsize,1)
        expected_q_value_batch  = reward_batch + self.gamma * next_target_q_value_batch * (1-done_batch)
        loss = nn.MSELoss()(q_value_batch , expected_q_value_batch)  
        self.optimizer.zero_grad()  
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters(): 
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  
    
    def save_model(self,path):
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), path+'checkpoint.pth')

    def load_model(self,path):
        self.target_net.load_state_dict(torch.load(path+'checkpoint.pth'))  
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)  
