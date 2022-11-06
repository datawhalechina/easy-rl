#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 01:23:07
LastEditor: JiangJi
LastEditTime: 2022-10-30 02:01:54
Discription: default parameters of QLearning
'''
from common.config import GeneralConfig,AlgoConfig

class GeneralConfigSarsa(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "CliffWalking-v0" # name of environment
        self.algo_name = "Sarsa" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 1 # random seed
        self.device = "cpu" # device to use
        self.train_eps = 400 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfigSarsa(AlgoConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 300 # epsilon decay rate
        self.gamma = 0.90 # discount factor
        self.lr = 0.1 # learning rate