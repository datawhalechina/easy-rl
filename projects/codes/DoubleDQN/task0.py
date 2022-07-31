#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-11-07 18:10:37
LastEditor: JiangJi
LastEditTime: 2022-07-21 21:52:31
Discription: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import gym
import torch
import datetime
import argparse

from common.utils import save_results,make_dir
from common.utils import plot_rewards,save_args
from DoubleDQN.double_dqn import DoubleDQN

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DoubleDQN',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--hidden_dim',default=256,type=int)
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
    n_actions = env.action_space.n
    agent = DoubleDQN(n_states,n_actions,cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0 # 记录一回合内的奖励
        state = env.reset() # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) 
            state = next_state 
            agent.update() 
            if done:
                break
        if i_ep % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep+1)%10 == 0: 
            print(f'Env:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)   
    print('Finish training!')
    return {'rewards':rewards,'ma_rewards':ma_rewards}

def test(cfg,env,agent):
    print('Start testing')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    
    for i_ep in range(cfg.test_eps):
        state = env.reset() 
        ep_reward = 0   
        while True:
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)  
            state = next_state  
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep+1}/{cfg.test_eps}, Reward:{ep_reward:.1f}")
    print('Finish testing!')
    return {'rewards':rewards,'ma_rewards':ma_rewards}

if __name__ == "__main__":
    cfg = get_args()
    print(cfg.device)
    # training
    env,agent = env_agent_config(cfg,seed=1)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    save_args(cfg)
    agent.save(path=cfg.model_path)
    save_results(res_dic, tag='train',
                 path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  
    # testing
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)
    res_dic = test(cfg,env,agent)
    save_results(res_dic, tag='test',
                 path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="test") 
