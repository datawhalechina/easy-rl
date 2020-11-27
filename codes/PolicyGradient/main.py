#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2020-11-24 19:52:40
Discription: 
Environment: 
'''
from itertools import count
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from env import env_init
from params import get_args
from agent import PolicyGradient
from params import SEQUENCE, SAVED_MODEL_PATH, RESULT_PATH
from utils import save_results,save_model
from plot import plot
def train(cfg):
    env,state_dim,n_actions = env_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    agent  = PolicyGradient(state_dim,device = device,lr = cfg.policy_lr)
    '''下面带pool都是存放的transition序列用于gradient'''
    state_pool = [] # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = [] 
    ''' 存储每个episode的reward用于绘图'''
    rewards = []
    moving_average_rewards = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir) # 使用tensorboard的writer
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', i_episode, ' Reward:',  ep_reward)
                break
        if i_episode > 0 and i_episode % cfg.batch_size == 0:
            agent.update(reward_pool,state_pool,action_pool)
            state_pool = [] # 每个episode的state
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode+1)
    writer.close()
    print('Complete training！')
    save_model(agent,model_path=SAVED_MODEL_PATH)
    '''存储reward等相关结果'''
    save_results(rewards,moving_average_rewards,tag='train',result_path=RESULT_PATH)
    plot(rewards)
    plot(moving_average_rewards,ylabel='moving_average_rewards_train')
        
def eval(cfg,saved_model_path = SAVED_MODEL_PATH):
    env,state_dim,n_actions = env_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    agent  = PolicyGradient(state_dim,device = device,lr = cfg.policy_lr)
    agent.load_model(saved_model_path+'checkpoint.pth')
    rewards = []
    moving_average_rewards = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/" + SEQUENCE
    writer = SummaryWriter(log_dir) # 使用tensorboard的writer
    for i_episode in range(cfg.eval_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward     
            state = next_state
            if done:
                print('Episode:', i_episode, ' Reward:',  ep_reward)
                break
        rewards.append(ep_reward)
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode+1)
    writer.close()
    print('Complete evaling！')
    
if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)
