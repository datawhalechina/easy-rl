#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 16:30:09
@LastEditor: John
@LastEditTime: 2020-06-14 11:38:42
@Discription: 
@Environment: python 3.7.7
'''
import matplotlib.pyplot as plt
import numpy as np
import os 

def plot(item,ylabel='rewards'):
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of DQN') 
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.savefig(os.path.dirname(__file__)+"/result/"+ylabel+".png")
    plt.show()
if __name__ == "__main__":

    output_path = os.path.dirname(__file__)+"/result/"
    rewards=np.load(output_path+"rewards.npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards.npy",)
    plot(rewards)
    plot(moving_average_rewards,ylabel='moving_average_rewards')
