#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2022-07-21 21:44:00
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import gym
import torch
import datetime
import argparse
from itertools import count

from pg import PolicyGradient
from common.utils import save_results, make_dir
from common.utils import plot_rewards


def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='PolicyGradient',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=300,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--lr',default=0.01,type=float,help="learning rate")
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--hidden_dim',default=36,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args


def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env_name)  
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    agent = PolicyGradient(n_states,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    state_pool = [] # temp states pool per several episodes
    action_pool = []
    reward_pool = [] 
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
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
                print(f'Episode：{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
                break
        if i_ep > 0 and i_ep % cfg.batch_size == 0:
            agent.update(reward_pool,state_pool,action_pool)
            state_pool = [] 
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Finish training!')
    env.close() # close environment
    return rewards, ma_rewards
            

def test(cfg,env,agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state = next_state
            if done:
                print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    env.close()
    return rewards, ma_rewards
    
if __name__ == "__main__":
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果

