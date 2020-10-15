#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2020-10-15 21:23:39
@Discription: 
@Environment: python 3.7.7
'''
from token import NUMBER
from typing import Sequence
import torch
import gym
from agent import DDPG
from env import NormalizedActions
from noise import OUNoise
import os
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--critic_lr", default=1e-3, type=float)  # critic学习率
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--memory_capacity", default=10000,
                        type=int, help="capacity of Replay Memory")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int)
    parser.add_argument("--train_steps", default=200, type=int)
    parser.add_argument("--eval_eps", default=200, type=int)  # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200,
                        type=int)  # 训练每个episode的长度
    parser.add_argument("--target_update", default=4, type=int,
                        help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()
    return config


def train(cfg):
    print('Start to train ! \n')
    env = NormalizedActions(gym.make("Pendulum-v0"))

    # 增加action噪声
    ou_noise = OUNoise(env.action_space)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPG(n_states, n_actions, device="cpu", critic_lr=1e-3,
                 actor_lr=1e-4, gamma=0.99, soft_tau=1e-2, memory_capacity=100000, batch_size=128)
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.train_eps+1):
        state = env.reset()
        ou_noise.reset()
        ep_reward = 0
        for i_step in range(1, cfg.train_steps+1):
            action = agent.select_action(state)
            action = ou_noise.get_action(
                action, i_step)  # 即paper中的random process
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    writer.close()
    print('Complete training！')
    ''' 保存模型 '''
    if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
        os.mkdir(SAVED_MODEL_PATH)
    agent.save_model(SAVED_MODEL_PATH+'checkpoint.pth')
    '''存储reward等相关结果'''
    if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
        os.mkdir(RESULT_PATH)
    np.save(RESULT_PATH+'rewards_train.npy', rewards)
    np.save(RESULT_PATH+'moving_average_rewards_train.npy', moving_average_rewards)
    np.save(RESULT_PATH+'steps_train.npy', ep_steps)

def eval(cfg, saved_model_path = SAVED_MODEL_PATH):
    print('start to eval ! \n')
    env = NormalizedActions(gym.make("Pendulum-v0"))
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states, n_actions, critic_lr=1e-3,
                 actor_lr=1e-4, gamma=0.99, soft_tau=1e-2, memory_capacity=100000, batch_size=128)
    agent.load_model(saved_model_path+'checkpoint.pth')
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.eval_eps+1):
        state = env.reset()  # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.eval_steps+1):
            action = agent.select_action(state)  # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)  # 更新环境参数
            ep_reward += reward
            state = next_state  # 跳转到下一个状态
            if done:
                break
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step, 'done: ', done)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    writer.close()
    '''存储reward等相关结果'''
    if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
        os.mkdir(RESULT_PATH)
    np.save(RESULT_PATH+'rewards_eval.npy', rewards)
    np.save(RESULT_PATH+'moving_average_rewards_eval.npy', moving_average_rewards)
    np.save(RESULT_PATH+'steps_eval.npy', ep_steps)

if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)
