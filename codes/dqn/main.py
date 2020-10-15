#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2020-10-15 22:00:28
@Discription: 
@Environment: python 3.7.7
'''
import gym
import torch
from agent import DQN
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''模型参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.95,
                        type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_decay", default=500, type=float)
    parser.add_argument("--policy_lr", default=0.01, type=float)
    parser.add_argument("--memory_capacity", default=1000,
                        type=int, help="capacity of Replay Memory") 

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int) # 训练的最大episode数目
    parser.add_argument("--train_steps", default=200, type=int)
    parser.add_argument("--target_update", default=2, type=int,
                        help="when(every default 2 eisodes) to update target net ") # 更新频率

    parser.add_argument("--eval_eps", default=100, type=int)  # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200,
                        type=int)  # 训练每个episode的长度
    config = parser.parse_args()

    return config
def train(cfg):
    print('Start to train ! \n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.train_eps+1):
        state = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.train_steps+1):
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
            state = next_state # 跳转到下一个状态
            agent.update() # 每步更新网络
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
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
    print('Complete training！')
    ''' 保存模型 '''
    if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
        os.mkdir(SAVED_MODEL_PATH)
    agent.save_model(SAVED_MODEL_PATH+'checkpoint.pth')
    print('model saved！')
    '''存储reward等相关结果'''
    save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path=RESULT_PATH)
    

def eval(cfg, saved_model_path = SAVED_MODEL_PATH):
    print('start to eval ! \n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
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
            action = agent.choose_action(state,train=False)  # 根据当前环境state选择action
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
    save_results(rewards,moving_average_rewards,ep_steps,tag='eval',result_path=RESULT_PATH)
    print('Complete evaling！')
    
if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)
