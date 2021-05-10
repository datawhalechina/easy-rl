#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-29 12:59:22
LastEditor: JiangJi
LastEditTime: 2021-05-06 16:58:01
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path


import gym
import torch
import datetime

from SAC.env import NormalizedActions
from SAC.agent import SAC
from common.utils import save_results, make_dir
from common.plot import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class SACConfig:
    def __init__(self) -> None:
        self.algo = 'SAC'
        self.env = 'Pendulum-v0'
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 300
        self.train_steps = 500
        self.eval_eps = 50
        self.eval_steps = 500
        self.gamma = 0.99
        self.mean_lambda=1e-3
        self.std_lambda=1e-3
        self.z_lambda=0.0
        self.soft_tau=1e-2
        self.value_lr  = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size  = 128
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg,seed=1):
    env = NormalizedActions(gym.make("Pendulum-v0"))
    env.seed(seed)
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    agent = SAC(state_dim,action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            if done:
                break
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        for i_step in range(cfg.eval_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        if (i_ep+1)%10==0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
    print('Complete evaling！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg=SACConfig()
    
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




