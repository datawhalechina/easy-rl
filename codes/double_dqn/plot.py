#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 16:30:09
@LastEditor: John
LastEditTime: 2020-09-01 22:46:43
@Discription: 
@Environment: python 3.7.7
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os 

def plot(item,ylabel='rewards',save_fig = True):
    sns.set()
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of DQN') 
    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    if save_fig:
        plt.savefig(os.path.dirname(__file__)+"/result/"+ylabel+".png")
    plt.show()
if __name__ == "__main__":

    output_path = os.path.dirname(__file__)+"/result/"
    rewards=np.load(output_path+"rewards.npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards.npy",)
    plot(rewards)
    plot(moving_average_rewards,ylabel='moving_average_rewards')
