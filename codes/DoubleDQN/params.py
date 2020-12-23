#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-12-22 15:22:17
LastEditor: John
LastEditTime: 2020-12-22 15:26:09
Discription: 
Environment: 
'''
import datetime
import os
import argparse

ALGO_NAME = 'Double DQN'
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''模型参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.95,
                        type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_decay", default=500, type=float)
    parser.add_argument("--policy_lr", default=0.01, type=float)
    parser.add_argument("--memory_capacity", default=1000,
                        type=int, help="capacity of Replay Memory") 

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int) # 训练的最大episode数目
    parser.add_argument("--train_steps", default=200, type=int)
    parser.add_argument("--target_update", default=2, type=int,
                        help="when(every default 2 eisodes) to update target net ") # 更新频率

    parser.add_argument("--eval_eps", default=100, type=int)  # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200,
                        type=int)  # 训练每个episode的长度
    config = parser.parse_args()

    return config