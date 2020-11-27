#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-23 13:44:52
LastEditor: John
LastEditTime: 2020-11-23 13:45:42
Discription: 
Environment: 
'''
import os
import numpy as np


def save_results(rewards,moving_average_rewards,tag='train',result_path='./result'):
    '''保存reward等结果
    '''
    if not os.path.exists(result_path): # 检测是否存在文件夹
        os.mkdir(result_path)
    np.save(result_path+'rewards_'+tag+'.npy', rewards)
    np.save(result_path+'moving_average_rewards_'+tag+'.npy', moving_average_rewards)
    print('results saved!')

def save_model(agent,model_path='./saved_model'):
    if not os.path.exists(model_path): # 检测是否存在文件夹
        os.mkdir(model_path)
    agent.save_model(model_path+'checkpoint.pth')
    print('model saved！')