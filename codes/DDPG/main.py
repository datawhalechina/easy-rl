#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-03-19 19:57:00
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
sys.path.append(os.getcwd()) # 添加当前终端路径
import torch
import gym
import numpy as np
import datetime
from DDPG.agent import DDPG
from DDPG.env import NormalizedActions,OUNoise
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): # 检测是否存在文件夹
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
    os.mkdir(RESULT_PATH)

class DDPGConfig:
    def __init__(self):
        self.gamma = 0.99
        self.critic_lr = 1e-3  
        self.actor_lr = 1e-4 
        self.memory_capacity = 10000
        self.batch_size = 128
        self.train_eps =300
        self.train_steps = 200
        self.eval_eps = 200
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 30
        self.soft_tau=1e-2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(cfg,env,agent):
    print('Start to train ! ')
    ou_noise = OUNoise(env.action_space) # action noise
    rewards = []
    ma_rewards = [] # moving average rewards
    ep_steps = []
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            action = agent.choose_action(state)
            action = ou_noise.get_action(
                action, i_step)  # 即paper中的random process
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        print('Episode:{}/{}, Reward:{}, Steps:{}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,i_step+1,done))
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = DDPGConfig()
    env = NormalizedActions(gym.make("Pendulum-v0"))
    env.seed(1) # 设置env随机种子
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)
    