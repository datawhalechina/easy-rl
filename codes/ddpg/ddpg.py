#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
@LastEditTime: 2020-06-14 11:43:17
@Discription: 
@Environment: python 3.7.7
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic
from memory import ReplayBuffer


class DDPG:
    def __init__(self, n_states, n_actions, hidden_dim=30, device="cpu", critic_lr=1e-3,
                 actor_lr=1e-4, gamma=0.99, soft_tau=1e-2, memory_capacity=100000, batch_size=128):
        self.device = device
        self.critic = Critic(n_states, n_actions, hidden_dim).to(device)
        self.actor = Actor(n_states, n_actions, hidden_dim).to(device)
        self.target_critic = Critic(n_states, n_actions, hidden_dim).to(device)
        self.target_actor = Actor(n_states, n_actions, hidden_dim).to(device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_criterion = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity)
        self.batch_size = batch_size
        self.soft_tau = soft_tau
        self.gamma = gamma

    def select_action(self, state):
        return self.actor.select_action(state)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size) 
        # 将所有变量转为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        
        value = self.critic(state, action)

        value_loss = self.critic_criterion(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
