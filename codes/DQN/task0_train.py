#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2021-09-15 02:19:54
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path

import gym
import torch
import datetime

from common.utils import save_results, make_dir
from common.plot import plot_rewards
from DQN.agent import DQN

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class DQNConfig:
    def __init__(self):
        self.algo = "DQN"  # 算法名称
        self.env = 'CartPole-v0' # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200 # 训练的回合数
        self.eval_eps = 30 # 测试的回合数
        self.gamma = 0.95 
        self.epsilon_start = 0.90 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500 # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # capacity of Replay Memory
        self.batch_size = 64
        self.target_update = 4 # 目标网络的更新频率
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # jian che
        self.hidden_dim = 256  # hidden size of net
        
def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env)  
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states,n_actions,cfg)
    return env,agent
    
def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
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
        if (i_ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep+1)%10 == 0:
            print('Episode:{}/{}, Reward:{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        # save ma_rewards
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg,env,agent):
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards = []  
    ma_rewards = [] # moving average rewards
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # reward per episode
        state = env.reset()  
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
    cfg = DQNConfig()

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
