#!/usr/bin/env python
# coding=utf-8
'''
Author: DingLi
Email: wangzhongren@sjtu.edu.cn
Date: 2022-10-31 22:54:00
LastEditor: DingLi
LastEditTime: 2022-11-14 10:43:18
Discription: CartPole-v1
'''

'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-10-26 07:50:24
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

class PER_DQN:
    def __init__(self,model,memory,cfg):

        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        ## copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        # self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.memory = memory 
        self.update_flag = False 
        
    def sample_action(self, state):
        ''' sample action with e-greedy policy
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action
    # @torch.no_grad()
    # def sample_action(self, state):
    #     ''' sample action with e-greedy policy
    #     '''
    #     self.sample_count += 1
    #     # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
    #     self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
    #         math.exp(-1. * self.sample_count / self.epsilon_decay) 
    #     if random.random() > self.epsilon:
    #         state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
    #         q_values = self.policy_net(state)
    #         action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
    #     else:
    #         action = random.randrange(self.n_actions)
    #     return action
    def predict_action(self,state):
        ''' predict action
        '''
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    def update(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            # print ("self.batch_size = ", self.batch_size)
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # sample a batch of transitions from replay buffer
        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), idxs_batch, is_weights_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)

        loss = torch.mean(torch.pow((q_value_batch - expected_q_value_batch) * torch.from_numpy(is_weights_batch).cuda(), 2))
        # loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  

        abs_errors = np.sum(np.abs(q_value_batch.cpu().detach().numpy() - expected_q_value_batch.cpu().detach().numpy()), axis=1)
        self.memory.batch_update(idxs_batch, abs_errors) 

        # backpropagation
        self.optimizer.zero_grad()  
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())  

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt",map_location=self.device)
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
