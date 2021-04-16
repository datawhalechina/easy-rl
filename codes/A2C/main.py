#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-04-05 11:14:39
@Discription: 
@Environment: python 3.7.9
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import datetime
from A2C.agent import A2C
from common.utils import save_results,make_dir,del_empty_dir



SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): 
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): 
    os.mkdir(RESULT_PATH)

class A2CConfig:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 3e-4 # learnning rate
        self.actor_lr = 1e-4 # learnning rate of actor network
        self.memory_capacity = 10000 # capacity of replay memory
        self.batch_size = 128
        self.train_eps = 200
        self.train_steps = 200
        self.eval_eps = 200
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim=256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def train(cfg,env,agent):
    print('Start to train ! ')
    for i_episode in range(cfg.train_eps):
        state = env.reset()
        log_probs = []
        values    = []
        rewards = []
        masks     = []
        entropy = 0
        ep_reward = 0
        for i_step in range(cfg.train_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = agent.model(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward+=reward
            state = next_state                                                  
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            if done:
                break
        print('Episode:{}/{}, Reward:{}, Steps:{}, Done:{}'.format(i_episode+1,cfg.train_eps,ep_reward,i_step+1,done))
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value =agent.model(next_state)
        returns = agent.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
 
    print('Complete training！')



if __name__ == "__main__":
    cfg = A2CConfig()
    env = gym.make('CartPole-v0')
    env.seed(1) # set random seed for env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2C(state_dim, action_dim, cfg)
    train(cfg,env,agent)

