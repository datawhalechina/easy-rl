#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2021-05-04 22:26:59
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import gym
import torch
import datetime
from DoubleDQN.agent import DoubleDQN
from common.plot import plot_rewards
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

class DoubleDQNConfig:
    def __init__(self):
        self.algo = "DoubleDQN" # name of algo
        self.env = 'CartPole-v0'  # env name
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200 # max tranng episodes
        self.eval_eps = 50 # max evaling episodes
        self.gamma = 0.95
        self.epsilon_start = 1 # start epsilon of e-greedy policy
        self.epsilon_end = 0.01 
        self.epsilon_decay = 500
        self.lr = 0.001 # learning rate
        self.memory_capacity = 100000 # capacity of Replay Memory
        self.batch_size = 64
        self.target_update = 2 # update frequency of target net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check gpu
        self.hidden_dim = 256 # hidden size of net
 
def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DoubleDQN(state_dim,action_dim,cfg)
    return env,agent
    
def train(cfg,env,agent):
    print('Start to train !')
    rewards,ma_rewards = [],[]
    for i_ep in range(cfg.train_eps):
        state = env.reset() 
        ep_reward = 0
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
        print(f'Episode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward},Epsilon:{agent.epsilon:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)   
    print('Complete training！')
    return rewards,ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []  
    ma_rewards = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset() 
        ep_reward = 0   
        while True:
            action = agent.predict(state)  
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
        print(f"Episode:{i_ep+1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")
    print('Complete evaling！')
    return rewards,ma_rewards    
    
if __name__ == "__main__":
    cfg = DoubleDQNConfig()
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
