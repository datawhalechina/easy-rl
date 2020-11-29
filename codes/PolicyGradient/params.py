#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:25:37
LastEditor: John
LastEditTime: 2020-11-26 19:11:21
Discription: 存储参数
Environment: 
'''
import argparse
import datetime
import os

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''训练参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--train_eps", default=300, type=int) # 训练的最大episode数目
    parser.add_argument("--eval_eps", default=100, type=int)  # 训练的最大episode数目
    parser.add_argument("--batch_size", default=4, type=int) # 用于gradient的episode数目
    parser.add_argument("--policy_lr", default=0.01, type=float) # 学习率
    config = parser.parse_args()
    return config