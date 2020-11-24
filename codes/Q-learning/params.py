#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-24 19:45:58
LastEditor: John
LastEditTime: 2020-11-24 19:53:13
Discription: 
Environment: 
'''
import argparse
import datetime
import os

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''训练的模型参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.9,
                        type=float, help="reward 的衰减率") 
    parser.add_argument("--epsilon_start", default=0.9,
                        type=float,help="e-greedy策略中初始epsilon")  
    parser.add_argument("--epsilon_end", default=0.1, type=float,help="e-greedy策略中的结束epsilon")
    parser.add_argument("--epsilon_decay", default=200, type=float,help="e-greedy策略中epsilon的衰减率")
    parser.add_argument("--policy_lr", default=0.1, type=float,help="学习率")
    parser.add_argument("--max_episodes", default=500, type=int,help="训练的最大episode数目") 

    config = parser.parse_args()

    return config