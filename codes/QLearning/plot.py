#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2020-10-07 21:00:29
Discription: 
Environment: 
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

def plot(item,ylabel='rewards'):
    sns.set()
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of Q-learning') 
    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    plt.savefig(os.path.dirname(__file__)+"/result/"+ylabel+".png")
    plt.show()

if __name__ == "__main__":

    output_path = os.path.dirname(__file__)+"/result/"
    rewards=np.load(output_path+"rewards_train.npy", )
    MA_rewards=np.load(output_path+"MA_rewards_train.npy")
    steps = np.load(output_path+"steps_train.npy")
    plot(rewards)
    plot(MA_rewards,ylabel='moving_average_rewards')
    plot(steps,ylabel='steps')