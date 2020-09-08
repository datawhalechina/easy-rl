#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 16:30:09
@LastEditor: John
LastEditTime: 2020-09-02 01:20:03
@Discription: 
@Environment: python 3.7.7
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; 
import numpy as np
import os 

def plot(item,ylabel='rewards',save_fig = True):
    '''plot using searborn to plot 
    '''
    sns.set()
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of DDPG') 
    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    plt.savefig(os.path.dirname(__file__)+"/result/"+ylabel+".png")
    plt.show()

# def plot(item,ylabel='rewards'):
#     
#     df = pd.DataFrame(dict(time=np.arange(len(item)),value=item))
#     g = sns.relplot(x="time", y="value", kind="line", data=df)
#     # g.fig.autofmt_xdate()
#     # sns.lineplot(time=time, data=item, color="r", condition="behavior_cloning")
#     # # sns.tsplot(time=time, data=x2, color="b", condition="dagger")
#     # plt.ylabel("Reward")
#     # plt.xlabel("Iteration Number")
#     # plt.title("Imitation Learning")

    # plt.show()
if __name__ == "__main__":

    output_path = os.path.dirname(__file__)+"/result/"
    rewards=np.load(output_path+"rewards.npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards.npy",)
    plot(rewards)
    plot(moving_average_rewards,ylabel='moving_average_rewards')
