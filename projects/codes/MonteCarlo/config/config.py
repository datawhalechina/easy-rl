#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-11-06 00:31:35
LastEditor: JiangJi
LastEditTime: 2022-11-06 00:45:44
Discription: parameters of MonteCarlo
'''
from common.config import GeneralConfig,AlgoConfig

class GeneralConfigMC(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "Racetrack-v0" # name of environment
        self.algo_name = "FirstVisitMC" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 1 # random seed
        self.device = "cpu" # device to use
        self.train_eps = 200 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfigMC(AlgoConfig):
    def __init__(self) -> None:
        self.gamma = 0.90 # discount factor
        self.epsilon = 0.15 # epsilon greedy
        self.lr = 0.1 # learning rate