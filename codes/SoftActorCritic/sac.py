#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-29 12:53:54
LastEditor: JiangJi
LastEditTime: 2021-12-22 15:41:19
Discription: 
Environment: 
'''
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(SoftQNet, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]
        
class SAC:
    def __init__(self,state_dim,action_dim,cfg) -> None:
        self.batch_size  = cfg.batch_size 
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.value_net  = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.target_value_net = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.soft_q_net = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)  
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)  
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
    def update(self, gamma=0.99,mean_lambda=1e-3,
        std_lambda=1e-3,
        z_lambda=0.0,
        soft_tau=1e-2,
        ):
        if len(self.memory) < self.batch_size:
            return 
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)


        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    def save(self, path):
        torch.save(self.value_net.state_dict(), path + "sac_value")
        torch.save(self.value_optimizer.state_dict(), path + "sac_value_optimizer")
        torch.save(self.soft_q_net.state_dict(), path + "sac_soft_q")
        torch.save(self.soft_q_optimizer.state_dict(), path + "sac_soft_q_optimizer")
        
        torch.save(self.policy_net.state_dict(), path + "sac_policy")
        torch.save(self.policy_optimizer.state_dict(), path + "sac_policy_optimizer")
        
    def load(self, path):
        self.value_net.load_state_dict(torch.load(path + "sac_value"))
        self.value_optimizer.load_state_dict(torch.load(path + "sac_value_optimizer"))
        self.target_value_net = copy.deepcopy(self.value_net)

        self.soft_q_net.load_state_dict(torch.load(path + "sac_soft_q"))
        self.soft_q_optimizer.load_state_dict(torch.load(path + "sac_soft_q_optimizer"))

        self.policy_net.load_state_dict(torch.load(path + "sac_policy"))
        self.policy_optimizer.load_state_dict(torch.load(path + "sac_policy_optimizer"))