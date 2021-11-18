#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2021-09-23 12:23:01
Discription: 
Environment: 
'''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties # 导入字体模块

def plot_rewards(rewards,ma_rewards,plot_cfg,tag='train'):
    sns.set() 
    plt.figure() # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(plot_cfg.device, plot_cfg.algo, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(ma_rewards,label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()

def plot_losses(losses,algo = "DQN",save=True,path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses,label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()
   
