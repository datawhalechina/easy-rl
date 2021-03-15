#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2021-03-13 14:56:50
@Discription: 
@Environment: python 3.7.7
'''
import sys,os
sys.path.append(os.getcwd()) # 添加当前终端路径
import gym
import torch
import datetime
from DQN.agent import DQN
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

class DQNConfig:
    def __init__(self):
        self.algo = "DQN" # 算法名称
        self.gamma = 0.99
        self.epsilon_start = 0.95 # e-greedy策略的初始epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.lr = 0.01 # 学习率
        self.memory_capacity = 800 # Replay Memory容量
        self.batch_size = 64
        self.train_eps = 250 # 训练的episode数目
        self.train_steps = 200 # 训练每个episode的最大长度
        self.target_update = 2 # target net的更新频率
        self.eval_eps = 20 # 测试的episode数目
        self.eval_steps = 200 # 测试每个episode的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
        self.hidden_dim = 128 # 神经网络隐藏层维度
 
def train(cfg,env,agent):
    print('Start to train !')
    rewards = []
    ma_rewards = [] # 滑动平均的reward
    ep_steps = []
    for i_episode in range(cfg.train_eps):
        state = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(cfg.train_steps):
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
        print('Episode:{}/{}, Reward:{}, Steps:{}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,i_step,done))
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)   
    print('Complete training！')
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg = DQNConfig()
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states,n_actions,cfg)
    rewards,ma_rewards = train(cfg,env,agent)
    agent.save(path=SAVED_MODEL_PATH)
    save_results(rewards,ma_rewards,tag='train',path=RESULT_PATH)
    plot_rewards(rewards,ma_rewards,tag="train",algo = cfg.algo,path=RESULT_PATH)
