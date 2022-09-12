#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:21:53
LastEditor: John
LastEditTime: 2022-08-27 00:04:08
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import gym
import torch
import datetime
import argparse
from itertools import count
import torch.nn.functional as F
from pg import PolicyGradient
from common.utils import save_results, make_dir,all_seed,save_args,plot_rewards
from common.models import MLP
from common.memories import PGReplay
from common.launcher import Launcher
from envs.register import register_env


class PGNet(MLP):
    ''' instead of outputing action, PG Net outputs propabilities of actions, we can use class inheritance from MLP here
    '''
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Main(Launcher):
    def get_args(self):
        """ Hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='PolicyGradient',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps") 
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
        parser.add_argument('--lr',default=0.01,type=float,help="learning rate")
        parser.add_argument('--update_fre',default=8,type=int)
        parser.add_argument('--hidden_dim',default=36,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=1,type=int,help="seed") 
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")           
        args = parser.parse_args()   
        default_args = {'result_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                        'model_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
        }
        args = {**vars(args),**default_args}  # type(dict)                         
        return args 
    def env_agent_config(self,cfg):
        register_env(cfg['env_name'])
        env = gym.make(cfg['env_name']) 
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg['seed'])
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n  # action dimension
        print(f"state dim: {n_states}, action dim: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        model = PGNet(n_states,1,hidden_dim=cfg['hidden_dim'])
        memory = PGReplay()
        agent = PolicyGradient(model,memory,cfg)
        return env,agent
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []
        for i_ep in range(cfg['train_eps']):
            state = env.reset()
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action = agent.sample_action(state) # sample action
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    reward = 0
                agent.memory.push((state,float(action),reward))
                state = next_state
                if done:
                    break
            if (i_ep+1) % 10 == 0:
                print(f"Episode：{i_ep+1}/{cfg['train_eps']}, Reward:{ep_reward:.2f}")
            if (i_ep+1) % cfg['update_fre'] == 0:
                agent.update()
            rewards.append(ep_reward)
        print('Finish training!')
        env.close() # close environment
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards}
        return res_dic
                
    def test(self,cfg,env,agent):
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []
        for i_ep in range(cfg['test_eps']):
            state = env.reset()
            ep_reward = 0
            for _ in range(cfg['ep_max_steps']):
                action = agent.predict_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    reward = 0
                state = next_state
                if done:
                    break
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}，Reward: {ep_reward:.2f}")
            rewards.append(ep_reward)         
        print("Finish testing!")
        env.close()
        return {'episodes':range(len(rewards)),'rewards':rewards}
    
if __name__ == "__main__":
    main = Main()
    main.run()


