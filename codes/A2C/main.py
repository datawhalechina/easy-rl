#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2020-11-08 22:19:56
@Discription: 
@Environment: python 3.7.9
'''
import torch
import gym
import os
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

from agent import A2C
from env import make_envs
from utils import SEQUENCE, SAVED_MODEL_PATH, RESULT_PATH
from utils import save_model,save_results

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.99,
                        type=float)  # reward 折扣因子
    parser.add_argument("--lr", default=3e-4, type=float)  # critic学习率
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--memory_capacity", default=10000,
                        type=int, help="capacity of Replay Memory")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=4000, type=int)
    parser.add_argument("--train_steps", default=5, type=int)
    parser.add_argument("--eval_eps", default=200, type=int)  # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200,
                        type=int)  # 训练每个episode的长度
    parser.add_argument("--target_update", default=4, type=int,
                        help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()
    return config

def test_env(agent,device='cpu'):
    env = gym.make("CartPole-v0")
    state = env.reset()
    ep_reward=0
    for _ in range(200):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, value = agent.model(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        ep_reward += reward
        if done:
            break
    return ep_reward


def train(cfg):
    print('Start to train ! \n')
    envs = make_envs(num_envs=16,env_name="CartPole-v0")
    n_states = envs.observation_space.shape[0]
    n_actions = envs.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = A2C(n_states, n_actions, hidden_dim=256)
    # moving_average_rewards = []
    # ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    state = envs.reset()
    for i_episode in range(1, cfg.train_eps+1):
        log_probs = []
        values    = []
        rewards = []
        masks     = []
        entropy = 0
        for i_step in range(1, cfg.train_steps+1):
            state = torch.FloatTensor(state).to(device)
            dist, value = agent.model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            state = next_state                                                  
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        if i_episode%20 == 0:
            print("reward",test_env(agent,device='cpu'))
        next_state = torch.FloatTensor(next_state).to(device)
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
    for _ in range(100):
        print("test_reward",test_env(agent,device='cpu'))
        
        # print('Episode:', i_episode, ' Reward: %i' %
        #       int(ep_reward[0]), 'n_steps:', i_step)
        # ep_steps.append(i_step)
        # rewards.append(ep_reward)
        # if i_episode == 1:
        #     moving_average_rewards.append(ep_reward[0])
        # else:
        #     moving_average_rewards.append(
        #         0.9*moving_average_rewards[-1]+0.1*ep_reward[0])
        # writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        # writer.add_scalar('steps_of_each_episode',
        #                   ep_steps[-1], i_episode)
    writer.close()
    print('Complete training！')
    ''' 保存模型 '''
    # save_model(agent,model_path=SAVED_MODEL_PATH)
    # '''存储reward等相关结果'''
    # save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path=RESULT_PATH)

# def eval(cfg, saved_model_path = SAVED_MODEL_PATH):
#     print('start to eval ! \n')
#     env = NormalizedActions(gym.make("Pendulum-v0"))
#     n_states = env.observation_space.shape[0]
#     n_actions = env.action_space.shape[0]
#     agent = DDPG(n_states, n_actions, critic_lr=1e-3,
#                  actor_lr=1e-4, gamma=0.99, soft_tau=1e-2, memory_capacity=100000, batch_size=128)
#     agent.load_model(saved_model_path+'checkpoint.pth')
#     rewards = []
#     moving_average_rewards = []
#     ep_steps = []
#     log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/" + SEQUENCE
#     writer = SummaryWriter(log_dir)
#     for i_episode in range(1, cfg.eval_eps+1):
#         state = env.reset()  # reset环境状态
#         ep_reward = 0
#         for i_step in range(1, cfg.eval_steps+1):
#             action = agent.choose_action(state)  # 根据当前环境state选择action
#             next_state, reward, done, _ = env.step(action)  # 更新环境参数
#             ep_reward += reward
#             state = next_state  # 跳转到下一个状态
#             if done:
#                 break
#         print('Episode:', i_episode, ' Reward: %i' %
#               int(ep_reward), 'n_steps:', i_step, 'done: ', done)
#         ep_steps.append(i_step)
#         rewards.append(ep_reward)
#         # 计算滑动窗口的reward
#         if i_episode == 1:
#             moving_average_rewards.append(ep_reward)
#         else:
#             moving_average_rewards.append(
#                 0.9*moving_average_rewards[-1]+0.1*ep_reward)
#         writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
#         writer.add_scalar('steps_of_each_episode',
#                           ep_steps[-1], i_episode)
#     writer.close()
#     '''存储reward等相关结果'''
#     if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
#         os.mkdir(RESULT_PATH)
#     np.save(RESULT_PATH+'rewards_eval.npy', rewards)
#     np.save(RESULT_PATH+'moving_average_rewards_eval.npy', moving_average_rewards)
#     np.save(RESULT_PATH+'steps_eval.npy', ep_steps)

if __name__ == "__main__":
    cfg = get_args()
    train(cfg)
    
    # cfg = get_args()
    # if cfg.train:
    #     train(cfg)
    #     eval(cfg)
    # else:
    #     model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
    #     eval(cfg,saved_model_path=model_path)
