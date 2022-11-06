#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2022-09-27 15:50:12
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import datetime
import gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from env import NormalizedActions,OUNoise
from ddpg import DDPG
from common.utils import all_seed
from common.memories import ReplayBufferQue
from common.launcher import Launcher
from envs.register import register_env

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
class Main(Launcher):
    def get_args(self):
        """ hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters") 
        parser.add_argument('--algo_name',default='DDPG',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='Pendulum-v1',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=300,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--max_steps',default=100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
        parser.add_argument('--critic_lr',default=1e-3,type=float,help="learning rate of critic")
        parser.add_argument('--actor_lr',default=1e-4,type=float,help="learning rate of actor")
        parser.add_argument('--memory_capacity',default=8000,type=int,help="memory capacity")
        parser.add_argument('--batch_size',default=128,type=int)
        parser.add_argument('--target_update',default=2,type=int)
        parser.add_argument('--tau',default=1e-2,type=float)
        parser.add_argument('--critic_hidden_dim',default=256,type=int)
        parser.add_argument('--actor_hidden_dim',default=256,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=1,type=int,help="random seed")
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")  
        args = parser.parse_args()   
        default_args = {'result_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                        'model_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
        }
        args = {**vars(args),**default_args}  # type(dict)                         
        return args

    def env_agent_config(self,cfg):
        register_env(cfg['env_name'])
        env = gym.make(cfg['env_name']) 
        env = NormalizedActions(env) # decorate with action noise
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg["seed"]) 
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        models = {"actor":Actor(n_states,n_actions,hidden_dim=cfg['actor_hidden_dim']),"critic":Critic(n_states,n_actions,hidden_dim=cfg['critic_hidden_dim'])}
        memories = {"memory":ReplayBufferQue(cfg['memory_capacity'])}
        agent = DDPG(models,memories,cfg)
        return env,agent
    def train(self,cfg, env, agent):
        print('Start training!')
        ou_noise = OUNoise(env.action_space)  # noise of action
        rewards = [] # record rewards for all episodes
        for i_ep in range(cfg['train_eps']):
            state = env.reset()
            ou_noise.reset()
            ep_reward = 0
            for i_step in range(cfg['max_steps']):
                action = agent.sample_action(state)
                action = ou_noise.get_action(action, i_step+1) 
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                agent.memory.push((state, action, reward, next_state, done))
                agent.update()
                state = next_state
                if done:
                    break
            if (i_ep+1)%10 == 0:
                print(f"Env:{i_ep+1}/{cfg['train_eps']}, Reward:{ep_reward:.2f}")
            rewards.append(ep_reward)
        print('Finish training!')
        return {'rewards':rewards}

    def test(self,cfg, env, agent):
        print('Start testing!')
        rewards = [] # record rewards for all episodes
        for i_ep in range(cfg['test_eps']):
            state = env.reset() 
            ep_reward = 0
            for i_step in range(cfg['max_steps']):
                action = agent.predict_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(ep_reward)
            print(f"Episode:{i_ep+1}/{cfg['test_eps']}, Reward:{ep_reward:.1f}")
        print('Finish testing!')
        return {'rewards':rewards}
if __name__ == "__main__":
    main = Main()
    main.run()

