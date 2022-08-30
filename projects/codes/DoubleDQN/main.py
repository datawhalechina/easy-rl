#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-11-07 18:10:37
LastEditor: JiangJi
LastEditTime: 2022-08-29 23:33:31
Discription: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import gym
import datetime
import argparse

from common.utils import all_seed
from common.models import MLP
from common.memories import ReplayBufferQue
from DoubleDQN.double_dqn import DoubleDQN
from common.launcher import Launcher
from envs.register import register_env
class Main(Launcher):
    def get_args(self):
        ''' hyperparameters
        '''
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='DoubleDQN',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
        parser.add_argument('--gamma',default=0.95,type=float,help="discounted factor")
        parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
        parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
        parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon")
        parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
        parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
        parser.add_argument('--batch_size',default=64,type=int)
        parser.add_argument('--target_update',default=4,type=int)
        parser.add_argument('--hidden_dim',default=256,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=1,type=int,help="seed") 
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
        args = parser.parse_args()   
        default_args = {'result_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                        'model_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
        }
        args = {**vars(args),**default_args}  # type(dict)                         
        return args
    def env_agent_config(self,cfg):
        ''' create env and agent
        '''  
        register_env(cfg['env_name'])
        env = gym.make(cfg['env_name']) 
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg["seed"]) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        models = {'Qnet':MLP(n_states,n_actions,hidden_dim=cfg['hidden_dim'])}
        memories = {'Memory':ReplayBufferQue(cfg['memory_capacity'])}
        agent = DoubleDQN(models,memories,cfg)
        return env,agent
        
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg["train_eps"]):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg['ep_max_steps']):
                action = agent.sample_action(state) 
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                agent.memory.push((state, action, reward, next_state, done)) 
                state = next_state 
                agent.update() 
                if done:
                    break
            if i_ep % cfg['target_update'] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            steps.append(ep_step)
            rewards.append(ep_reward)
            if (i_ep+1)%10 == 0: 
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}: Epislon: {agent.epsilon:.3f}')
        print("Finish training!")
        env.close()
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
        return res_dic

    def test(self,cfg,env,agent):
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg['test_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state 
            for _ in range(cfg['ep_max_steps']):
                action = agent.predict_action(state) 
                next_state, reward, done, _ = env.step(action)  
                state = next_state  
                ep_reward += reward
                if done:
                    break
            steps.append(ep_step)
            rewards.append(ep_reward)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}，Reward: {ep_reward:.2f}")
        print("Finish testing!")
        env.close()
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
