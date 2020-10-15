#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 16:30:09
@LastEditor: John
LastEditTime: 2020-10-15 22:01:50
@Discription: 
@Environment: python 3.7.7
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

def plot(item,ylabel='rewards_train', save_fig = True):
    '''plot using searborn to plot 
    '''
    sns.set()
    plt.figure()
    plt.plot(np.arange(len(item)), item)
    plt.title(ylabel+' of DQN') 
    plt.ylabel(ylabel)
    plt.xlabel('episodes')
    if save_fig:
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

    output_path = os.path.split(os.path.abspath(__file__))[0]+"/result/"
    tag = 'train'
    rewards=np.load(output_path+"rewards_"+tag+".npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards_"+tag+".npy",)
    steps=np.load(output_path+"steps_"+tag+".npy")
    plot(rewards)
    plot(moving_average_rewards,ylabel='moving_average_rewards_'+tag)
    plot(steps,ylabel='steps_'+tag)
    tag = 'eval'
    rewards=np.load(output_path+"rewards_"+tag+".npy", )
    moving_average_rewards=np.load(output_path+"moving_average_rewards_"+tag+".npy",)
    steps=np.load(output_path+"steps_"+tag+".npy")
    plot(rewards,ylabel='rewards_'+tag)
    plot(moving_average_rewards,ylabel='moving_average_rewards_'+tag)
    plot(steps,ylabel='steps_'+tag) 
