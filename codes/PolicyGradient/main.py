#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2020-11-23 12:06:15
Discription: 
Environment: 
'''
from itertools import count
import torch
from env import env_init
from params import get_args
from agent import PolicyGradient

def train(cfg):
    env,n_states,n_actions = env_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    agent  = PolicyGradient(n_states,device = device,lr = cfg.policy_lr)
    '''下面带pool都是存放的transition序列用于gradient'''
    state_pool = [] # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = [] 
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for t in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', i_episode, ' Reward:',  ep_reward)
                break
        # if i_episode % cfg.batch_size == 0:
        if i_episode > 0 and i_episode % 5 == 0:
            agent.update(reward_pool,state_pool,action_pool)
            state_pool = [] # 每个episode的state
            action_pool = []
            reward_pool = []
        
        
if __name__ == "__main__":
    cfg = get_args()
    train(cfg)