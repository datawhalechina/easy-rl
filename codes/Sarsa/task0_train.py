#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2021-05-06 17:12:37
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import datetime
from envs.racetrack_env import RacetrackEnv
from Sarsa.agent import Sarsa
from common.plot import plot_rewards
from common.utils import save_results,make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class SarsaConfig:
    ''' parameters for Sarsa
    '''
    def __init__(self):
        self.algo = 'Qlearning'
        self.env = 'CliffWalking-v0' # 0 up, 1 right, 2 down, 3 left
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200
        self.test_eps = 50
        self.epsilon = 0.15 # epsilon: The probability to select a random action . 
        self.gamma = 0.9 # gamma: Gamma discount factor.
        self.lr = 0.2 # learning rate: step size parameter
        self.n_steps = 2000

def env_agent_config(cfg,seed=1):
    env = RacetrackEnv()
    action_dim=9
    agent = Sarsa(action_dim,cfg)
    return env,agent
        
def train(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
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
        if (i_episode+1)%10==0:
            print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.train_eps,ep_reward))
    return rewards,ma_rewards

def eval(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.test_eps):
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
            state = next_state
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode+1)%10==0:
            print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.test_eps,ep_reward))
    print('Complete evaling！')
    return rewards,ma_rewards
        
if __name__ == "__main__":
    cfg = SarsaConfig()
    env,agent = env_agent_config(cfg,seed=1)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(cfg.result_path,cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)

    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
    
    

