#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2021-04-13 18:49:44
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import gym
import torch
import datetime
from DQN.agent import DQN
from common.plot import plot_rewards
from common.utils import save_results,make_dir,del_empty_dir

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = curr_path+"/saved_model/"+SEQUENCE+'/' # path to save model
RESULT_PATH = curr_path+"/results/"+SEQUENCE+'/' # path to save rewards
make_dir(curr_path+"/saved_model/",curr_path+"/results/")
del_empty_dir(curr_path+"/saved_model/",curr_path+"/results/")

class DQNConfig:
    def __init__(self):
        self.env = 'LunarLander-v2'
        self.algo = "DQN"  # name of algo
        self.gamma = 0.95
        self.epsilon_start = 1 # e-greedy策略的初始epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001 # learning rate
        self.memory_capacity = 1000000 # Replay Memory容量
        self.batch_size = 64
        self.train_eps = 300 # 训练的episode数目
        self.train_steps = 1000
        self.target_update = 2 # target net的更新频率
        self.eval_eps = 20 # 测试的episode数目
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
        self.hidden_dim = 256 # 神经网络隐藏层维度
 
def train(cfg,env,agent):
    print('Start to train !')
    rewards = []
    ma_rewards = [] # moveing average reward
    for i_episode in range(cfg.train_eps):
        state = env.reset() 
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action) 
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) 
            state = next_state 
            agent.update()
            if done:
                break 
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:{}/{}, Reward:{}'.format(i_episode+1,cfg.train_eps,ep_reward))
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)   
    print('Complete training！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = DQNConfig()
    env = gym.make(cfg.env)
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim,action_dim,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(SAVED_MODEL_PATH,RESULT_PATH)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)
    del_empty_dir(SAVED_MODEL_PATH,RESULT_PATH)