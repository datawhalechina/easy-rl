#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2021-03-12 17:01:43
Discription: 
Environment: 
'''
import sys,os
sys.path.append(os.getcwd())
import datetime
from envs.racetrack_env import RacetrackEnv
from Sarsa.agent import Sarsa
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

class SarsaConfig:
    ''' parameters for Sarsa
    '''
    def __init__(self):
       self.epsilon = 0.15 # epsilon: The probability to select a random action . 
       self.gamma = 0.9 # gamma: Gamma discount factor.
       self.lr = 0.2 # learning rate: step size parameter
       self.n_episodes = 150
       self.n_steps = 2000

def sarsa_train(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.n_episodes):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        state = env.reset()
        ep_reward = 0
        while True:
        # for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action,done)
            state = next_state
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        # if (i_episode+1)%10==0:
        #     print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.n_episodes,ep_reward))
    return rewards,ma_rewards
    
if __name__ == "__main__":
    sarsa_cfg = SarsaConfig()
    env = RacetrackEnv()
    n_actions=9
    agent = Sarsa(n_actions,sarsa_cfg)
    rewards,ma_rewards = sarsa_train(sarsa_cfg,env,agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = "On-Policy First-Visit MC Control",path=RESULT_PATH)


