#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 10:01:09
@LastEditor: John
LastEditTime: 2021-04-05 11:06:23
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
from DQN_cnn.env import get_screen
from DQN_cnn.agent import DQNcnn
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = curr_path+"/saved_model/"+SEQUENCE+'/' # path to save model
if not os.path.exists(curr_path+"/saved_model/"): 
    os.mkdir(curr_path+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): 
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = curr_path+"/results/"+SEQUENCE+'/' # path to save rewards
if not os.path.exists(curr_path+"/results/"): 
    os.mkdir(curr_path+"/results/")
if not os.path.exists(RESULT_PATH): 
    os.mkdir(RESULT_PATH)

class DQNcnnConfig:
    def __init__(self) -> None:
        self.algo = "DQN_cnn"  # name of algo
        self.gamma = 0.99
        self.epsilon_start = 0.95  # e-greedy策略的初始epsilon
        self.epsilon_end = 0.05
        self.epsilon_decay = 200
        self.lr = 0.01  # leanring rate
        self.memory_capacity = 10000  # Replay Memory容量
        self.batch_size = 64
        self.train_eps = 250  # 训练的episode数目
        self.train_steps = 200  # 训练每个episode的最大长度
        self.target_update = 4  # target net的更新频率
        self.eval_eps = 20  # 测试的episode数目
        self.eval_steps = 200  # 测试每个episode的最大长度
        self.hidden_dim = 128  # 神经网络隐藏层维度
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used

def train(cfg, env, agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, cfg.device)
        current_screen = get_screen(env, cfg.device)
        state = current_screen - last_screen
        ep_reward = 0
        for i_step in range(cfg.train_steps+1):
            # Select and perform an action
            action = agent.choose_action(state)
            _, reward, done, _ = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward], device=cfg.device)
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, cfg.device)
            if done:
                break
            state_ = current_screen - last_screen
            # Store the transition in memory
            agent.memory.push(state, action, state_, reward)
            # Move to the next state
            state = state_
            # Perform one step of the optimization (on the target network)
            agent.update()
        # Update the target network, copying all weights and biases in DQN
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:{}/{}, Reward:{}, Steps:{}, Explore:{:.2f}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,i_step+1,agent.epsilon,done))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    return rewards,ma_rewards


if __name__ == "__main__":
    cfg = DQNcnnConfig()
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen(env,device)
    # 因为这里环境的state需要从默认的向量改为图像，所以要unwrapped更改state
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    init_screen = get_screen(env, cfg.device)
    _, _, screen_height, screen_width = init_screen.shape
    # Get number of actions from gym action space
    action_dim = env.action_space.n
    agent = DQNcnn(screen_height, screen_width,
                   action_dim, cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)
