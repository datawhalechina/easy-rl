#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-15 21:31:19
LastEditor: John
LastEditTime: 2020-10-15 21:31:25
Discription: 
Environment: 
'''
import os
import numpy as np
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def save_results(rewards,moving_average_rewards,ep_steps,path=RESULT_PATH):
    if not os.path.exists(path): # 检测是否存在文件夹
            os.mkdir(path)
    np.save(RESULT_PATH+'rewards_train.npy', rewards)
    np.save(RESULT_PATH+'moving_average_rewards_train.npy', moving_average_rewards)
    np.save(RESULT_PATH+'steps_train.npy',ep_steps )