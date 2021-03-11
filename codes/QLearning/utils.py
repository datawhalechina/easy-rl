#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-23 13:44:52
LastEditor: John
LastEditTime: 2021-03-11 19:18:34
Discription: 
Environment: 
'''
import os
import numpy as np


def save_results(rewards,tag='train',result_path='./result'):
    '''保存reward等结果
    '''
    if not os.path.exists(result_path): # 检测是否存在文件夹
        os.mkdir(result_path)
    np.save(result_path+'rewards_'+tag+'.npy', rewards)
    print('results saved!')
