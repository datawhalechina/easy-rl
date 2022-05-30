#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2022-04-29 20:18:13
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # current path of file
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import datetime
import torch
from envs.racetrack_env import RacetrackEnv
from Sarsa.sarsa import Sarsa
from common.utils import save_results,make_dir,plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class Config:
    ''' parameters for Sarsa
    '''
    def __init__(self):
        self.algo_name = 'Qlearning'
        self.env_name = 'CliffWalking-v0' # 0 up, 1 right, 2 down, 3 left
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check GPU
        self.result_path = curr_path+"/outputs/" +self.env_name+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env_name+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 300 # training episodes
        self.test_eps = 20 # testing episodes
        self.n_steps = 200 # maximum steps per episode 
        self.epsilon_start = 0.90  # start value of epsilon
        self.epsilon_end = 0.01  # end value of epsilon
        self.epsilon_decay = 200  # decay rate of epsilon
        self.gamma = 0.99 # gamma: Gamma discount factor.
        self.lr = 0.2 # learning rate: step size parameter 
        self.save = True # if save figures

def env_agent_config(cfg,seed=1):
    env = RacetrackEnv()
    n_states = 9 # number of actions
    agent = Sarsa(n_states,cfg)
    return env,agent
        
def train(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        action = agent.choose_action(state)
        ep_reward = 0
        # while True:
        for _ in range(cfg.n_steps):
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action,done)
            state = next_state
            action = next_action
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_ep+1)%2==0:
            print(f"Episode:{i_ep+1}, Reward:{ep_reward}, Epsilon:{agent.epsilon}")
    return rewards,ma_rewards

def test(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        state = env.reset()
        ep_reward = 0
        while True:
        # for _ in range(cfg.n_steps):
            action = agent.predict_action(state)
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            state = next_state
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_ep+1)%1==0:
            print("Episode:{}/{}: Reward:{}".format(i_ep+1, cfg.test_eps,ep_reward))
    print('Complete testing！')
    return rewards,ma_rewards
        
if __name__ == "__main__":
    cfg = Config()
    env,agent = env_agent_config(cfg,seed=1)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(cfg.result_path,cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")

    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = test(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='test',path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="test")
    
    

