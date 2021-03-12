#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 14:26:44
LastEditor: John
LastEditTime: 2021-03-12 16:15:46
Discription: 
Environment: 
'''
import sys,os
sys.path.append(os.getcwd())
import argparse
import datetime

from envs.racetrack_env import RacetrackEnv
from MonteCarlo.agent import FisrtVisitMC
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
    os.mkdir(RESULT_PATH)

class MCConfig:
    def __init__(self): 
        self.epsilon = 0.15 # epsilon: The probability to select a random action . 
        self.gamma = 0.9 # gamma: Gamma discount factor.
        self.n_episodes = 300
        self.n_steps = 2000

def get_mc_args():
    '''set parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", default=0.15, type=float)  # epsilon: The probability to select a random action . float between 0 and 1.
    parser.add_argument("--gamma", default=0.9, type=float)  # gamma: Gamma discount factor.
    parser.add_argument("--n_episodes", default=150, type=int)
    parser.add_argument("--n_steps", default=2000, type=int)
    mc_cfg = parser.parse_args()
    return mc_cfg



def mc_train(cfg,env,agent):
    rewards = []
    ma_rewards = [] # moving average rewards
    for i_episode in range(cfg.n_episodes):
        one_ep_transition = []
        state = env.reset()
        ep_reward = 0
        # while True:
        for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            one_ep_transition.append((state, action, reward))
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        agent.update(one_ep_transition)
        if (i_episode+1)%10==0:
            print("Episode:{}/{}: Reward:{}".format(i_episode+1, mc_cfg.n_episodes,ep_reward))
    return rewards,ma_rewards
if __name__ == "__main__":
    mc_cfg = MCConfig()
    env = RacetrackEnv()
    n_actions=9
    agent = FisrtVisitMC(n_actions,mc_cfg)
    rewards,ma_rewards= mc_train(mc_cfg,env,agent)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = "On-Policy First-Visit MC Control",path=RESULT_PATH)
    

