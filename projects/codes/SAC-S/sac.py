import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
class SAC:
    def __init__(self,n_actions,models,memory,cfg):
        self.device = cfg.device
        self.value_net  = models['ValueNet'].to(self.device) # $\psi$
        self.target_value_net = models['ValueNet'].to(self.device) # $\bar{\psi}$
        self.soft_q_net = models['SoftQNet'].to(self.device) # $\theta$
        self.policy_net = models['PolicyNet'].to(self.device) # $\phi$
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)  
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
    def update(self):
        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
