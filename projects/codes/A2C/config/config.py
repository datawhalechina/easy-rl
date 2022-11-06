#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 00:53:03
LastEditor: JiangJi
LastEditTime: 2022-11-01 00:17:55
Discription: default parameters of A2C
'''
from common.config import GeneralConfig,AlgoConfig

class GeneralConfigA2C(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "A2C" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 1 # random seed
        self.device = "cuda" # device to use
        self.train_eps = 1000 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfigA2C(AlgoConfig):
    def __init__(self) -> None:
        self.continuous = False # continuous or discrete action space
        self.hidden_dim = 256 # hidden_dim for MLP
        self.gamma = 0.99 # discount factor
        self.actor_lr = 3e-4 # learning rate of actor
        self.critic_lr = 1e-3 # learning rate of critic
        self.actor_hidden_dim = 256 # hidden_dim for actor MLP
        self.critic_hidden_dim = 256 # hidden_dim for critic MLP
        self.buffer_size = 100000 # size of replay buffer
        self.batch_size = 64 # batch size