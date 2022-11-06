#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-08-16 23:05:25
LastEditor: JiangJi
LastEditTime: 2022-11-01 00:33:49
Discription: 
'''
import torch
import numpy as np
from torch.distributions import Categorical,Normal

        
class A2C:
    def __init__(self,models,memories,cfg):
        self.n_actions = cfg.n_actions
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device) 
        self.continuous = cfg.continuous
        if hasattr(cfg,'action_bound'):
            self.action_bound = cfg.action_bound
        self.memory = memories['ACMemory']
        self.actor = models['Actor'].to(self.device)
        self.critic = models['Critic'].to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
    def sample_action(self,state):
        # state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        # dist = self.actor(state)
        # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
        # value = self.critic(state) # note that 'dist' need require_grad=True
        # self.value = value.detach().cpu().numpy().squeeze(0)[0]
        # action = np.random.choice(self.n_actions, p=dist.detach().cpu().numpy().squeeze(0)) # shape(p=(n_actions,1)
        # self.log_prob = torch.log(dist.squeeze(0)[action])
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1,), sigma.view(1,))
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            self.value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            self.log_prob = dist.log_prob(action).squeeze(dim=0) # Tensor([0.])
            self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().detach().numpy()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample() # Tensor([0])
            value = self.critic(state)
            self.value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            self.log_prob = dist.log_prob(action).squeeze(dim=0) # Tensor([0.])
            self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1,), sigma.view(1,))
            action = dist.sample()
            return action.cpu().detach().numpy()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            dist = self.actor(state)
            # value = self.critic(state) # note that 'dist' need require_grad=True
            # value = value.detach().cpu().numpy().squeeze(0)[0]
            action = np.random.choice(self.n_actions, p=dist.detach().cpu().numpy().squeeze(0)) # shape(p=(n_actions,1)
            return action
    def update(self,next_state,entropy):
        value_pool,log_prob_pool,reward_pool = self.memory.sample()
        value_pool = torch.tensor(value_pool, device=self.device)
        log_prob_pool = torch.stack(log_prob_pool)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        next_value = self.critic(next_state)
        returns = np.zeros_like(reward_pool)
        for t in reversed(range(len(reward_pool))):
            next_value = reward_pool[t] + self.gamma * next_value # G(s_{t},a{t}) = r_{t+1} + gamma * V(s_{t+1})
            returns[t] = next_value
        returns = torch.tensor(returns, device=self.device)
        advantages = returns - value_pool
        actor_loss = (-log_prob_pool * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        tot_loss = actor_loss + critic_loss + 0.001 * entropy
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        tot_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        self.memory.clear()
    def save_model(self, path):
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_checkpoint.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_checkpoint.pt")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor_checkpoint.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/critic_checkpoint.pt"))