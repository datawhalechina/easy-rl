#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:25:37
LastEditor: John
LastEditTime: 2020-11-22 23:32:44
Discription: 存储参数
Environment: 
'''
import argparse
def get_args():
    '''训练参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_eps", default=1200, type=int) # 训练的最大episode数目
    parser.add_argument("--policy_lr", default=0.01, type=float) # 学习率
    config = parser.parse_args()
    return config