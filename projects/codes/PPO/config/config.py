#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 11:30:56
LastEditor: JiangJi
LastEditTime: 2022-10-31 00:33:15
Discription: default parameters of PPO
'''
from common.config import GeneralConfig,AlgoConfig

class GeneralConfigPPO(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v0"
        self.algo_name = "PPO"
        self.seed = 1
        self.device = "cuda"
        self.train_eps = 100 # number of episodes for training
        self.test_eps = 10 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode

class AlgoConfigPPO(AlgoConfig):
    def __init__(self) -> None:
        self.gamma = 0.99 # discount factor
        self.continuous = False # continuous action space or not
        self.policy_clip = 0.2 # clip range of policy
        self.n_epochs = 10 # number of epochs
        self.gae_lambda = 0.95 # gae lambda
        self.actor_lr = 0.0003 # learning rate of actor
        self.critic_lr = 0.0003 # learning rate of critic
        self.actor_hidden_dim = 256 # 
        self.critic_hidden_dim = 256
        self.n_epochs = 4 # epochs 
        self.batch_size = 5 # 
        self.policy_clip = 0.2
        self.update_fre = 20 # frequency of updating agent
