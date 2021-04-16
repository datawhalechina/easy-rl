#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-20 17:43:17
LastEditor: John
LastEditTime: 2021-04-05 11:19:20
Discription: 
Environment: 
'''
import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


learning_rate = 3e-4

# Constants
GAMMA = 0.99

class A2CConfig:
    ''' hyperparameters
    '''
    def __init__(self):
        self.gamma = 0.99
        self.lr = 3e-4 # learnning rate
        self.actor_lr = 1e-4 # learnning rate of actor network
        self.memory_capacity = 10000 # capacity of replay memory
        self.batch_size = 128
        self.train_eps = 3000
        self.train_steps = 200
        self.eval_eps = 200
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.n_actions = n_actions
        self.critic_linear1 = nn.Linear(n_states, hidden_dim)
        self.critic_linear2 = nn.Linear(hidden_dim, 1)

        self.actor_linear1 = nn.Linear(n_states, hidden_dim)
        self.actor_linear2 = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

class A2C:
    def __init__(self,n_states,n_actions,cfg):
        self.model = ActorCritic(n_states, n_actions, cfg.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
    def choose_action(self,state):
        pass
    def update(self):
        pass
    
def train(cfg,env,agent):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    actor_critic = ActorCritic(n_states, n_actions, cfg.hidden_dim)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(cfg.train_eps):
        log_probs = []
        values = []
        rewards = []
        state = env.reset()
        for steps in range(cfg.train_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(n_actions, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            if done or steps == cfg.train_steps-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps+1, average_lengths[-1]))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        
    
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

if __name__ == "__main__":
    cfg = A2CConfig()
    env = gym.make("CartPole-v0")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2C(n_states,n_actions,cfg)
    train(cfg,env,agent)    