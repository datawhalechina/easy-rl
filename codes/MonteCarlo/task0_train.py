#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 14:26:44
LastEditor: John
LastEditTime: 2021-05-05 17:27:50
Discription: 
Environment: 
'''

import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import torch
import datetime

from common.utils import save_results,make_dir
from common.plot import plot_rewards
from MonteCarlo.agent import FisrtVisitMC
from envs.racetrack_env import RacetrackEnv

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class MCConfig:
    def __init__(self):
        self.algo = "MC"  # name of algo
        self.env = 'Racetrack'
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # path to save models
        # epsilon: The probability to select a random action .
        self.epsilon = 0.15
        self.gamma = 0.9  # gamma: Gamma discount factor.
        self.train_eps = 200
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu

def env_agent_config(cfg,seed=1):
    env = RacetrackEnv()
    action_dim = 9
    agent = FisrtVisitMC(action_dim, cfg)
    return env,agent
    
def train(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        one_ep_transition = []
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
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
        if (i_ep+1) % 10 == 0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}: Reward:{ep_reward}")
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 10 == 0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}: Reward:{ep_reward}")
    return rewards, ma_rewards
    
if __name__ == "__main__":
    cfg = MCConfig()
    
    # train
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="train",
                 algo=cfg.algo, path=cfg.result_path)
    # eval
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="eval",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
