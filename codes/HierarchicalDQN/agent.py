#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-24 22:18:18
LastEditor: John
LastEditTime: 2021-05-04 22:39:34
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import numpy as np
import random,math
import torch.optim as optim
from common.model import MLP
from common.memory import ReplayBuffer

class HierarchicalDQN:
    def __init__(self,state_dim,action_dim,cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.frame_idx = 0 
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end ) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.policy_net = MLP(2*state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.meta_policy_net = MLP(state_dim, state_dim,cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=cfg.lr)
        self.meta_optimizer = optim.Adam(self.meta_policy_net.parameters(),lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.meta_memory = ReplayBuffer(cfg.memory_capacity)
        self.loss_numpy  = 0
        self.meta_loss_numpy  = 0
        self.losses = []
        self.meta_losses = []
    def to_onehot(self,x):
        oh = np.zeros(self.state_dim)
        oh[x - 1] = 1.
        return oh
    def set_goal(self,state):
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                goal = self.meta_policy_net(state).max(1)[1].item() 
        else:
            goal = random.randrange(self.state_dim)
        return goal
    def choose_action(self,state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_value = self.policy_net(state)
                action = q_value.max(1)[1].item()  
        else:
            action = random.randrange(self.action_dim)
        return action
    def update(self):
        self.update_policy()
        self.update_meta()
    def update_policy(self): 
        if self.batch_size > len(self.memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch,device=self.device,dtype=torch.float)
        action_batch = torch.tensor(action_batch,device=self.device,dtype=torch.int64).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch,device=self.device,dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch,device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch),device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch).squeeze(1)
        next_state_values = self.policy_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + 0.99 * next_state_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values) 
        self.optimizer.zero_grad() 
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  
        self.loss_numpy = loss.detach().cpu().numpy()
        self.losses.append(self.loss_numpy)  
    def update_meta(self):
        if self.batch_size > len(self.meta_memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.meta_memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch,device=self.device,dtype=torch.float)
        action_batch = torch.tensor(action_batch,device=self.device,dtype=torch.int64).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch,device=self.device,dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch,device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch),device=self.device)
        q_values = self.meta_policy_net(state_batch).gather(dim=1, index=action_batch).squeeze(1)
        next_state_values = self.meta_policy_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + 0.99 * next_state_values * (1-done_batch)
        meta_loss = nn.MSELoss()(q_values, expected_q_values) 
        self.meta_optimizer.zero_grad() 
        meta_loss.backward()
        for param in self.meta_policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step() 
        self.meta_loss_numpy = meta_loss.detach().cpu().numpy()
        self.meta_losses.append(self.meta_loss_numpy)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path+'policy_checkpoint.pth')
        torch.save(self.meta_policy_net.state_dict(), path+'meta_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path+'policy_checkpoint.pth'))
        self.meta_policy_net.load_state_dict(torch.load(path+'meta_checkpoint.pth'))
        

        
        