import torch
import numpy as np


        
class A2C:
    def __init__(self,models,memories,cfg):
        self.n_actions = cfg['n_actions']
        self.gamma = cfg['gamma']
        self.device = torch.device(cfg['device']) 
        self.memory = memories['ACMemory']
        self.actor = models['Actor'].to(self.device)
        self.critic = models['Critic'].to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
    def sample_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        dist = self.actor(state)
        value = self.critic(state) # note that 'dist' need require_grad=True
        value = value.detach().numpy().squeeze(0)[0]
        action = np.random.choice(self.n_actions, p=dist.detach().numpy().squeeze(0)) # shape(p=(n_actions,1)
        return action,value,dist 
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        dist = self.actor(state)
        value = self.critic(state) # note that 'dist' need require_grad=True
        value = value.detach().numpy().squeeze(0)[0]
        action = np.random.choice(self.n_actions, p=dist.detach().numpy().squeeze(0)) # shape(p=(n_actions,1)
        return action,value,dist 
    def update(self,next_state,entropy):
        value_pool,log_prob_pool,reward_pool = self.memory.sample()
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        next_value = self.critic(next_state)
        returns = np.zeros_like(reward_pool)
        for t in reversed(range(len(reward_pool))):
            next_value = reward_pool[t] + self.gamma * next_value # G(s_{t},a{t}) = r_{t+1} + gamma * V(s_{t+1})
            returns[t] = next_value
        returns = torch.tensor(returns, device=self.device)
        value_pool = torch.tensor(value_pool, device=self.device)
        advantages = returns - value_pool
        log_prob_pool = torch.stack(log_prob_pool)
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