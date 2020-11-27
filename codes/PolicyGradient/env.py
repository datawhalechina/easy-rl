#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:23:10
LastEditor: John
LastEditTime: 2020-11-23 11:55:24
Discription: 
Environment: 
'''
import gym

def env_init():
    env = gym.make('CartPole-v0') # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    return env,state_dim,n_actions