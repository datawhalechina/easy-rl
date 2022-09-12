#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2022-08-26 23:03:39
Discription: 
Environment: 
'''
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path
import gym
import datetime
import argparse
from envs.register import register_env
from envs.wrappers import CliffWalkingWapper
from Sarsa.sarsa import Sarsa
from common.utils import all_seed
from common.launcher import Launcher

class Main(Launcher):
    def get_args(self):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default = 'Sarsa',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default = 'Racetrack-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default = 300,type=int,help="episodes of training") 
        parser.add_argument('--test_eps',default = 20,type=int,help="episodes of testing") 
        parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps") 
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor") 
        parser.add_argument('--epsilon_start',default=0.90,type=float,help="initial value of epsilon") 
        parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon") 
        parser.add_argument('--epsilon_decay',default=200,type=int,help="decay rate of epsilon") 
        parser.add_argument('--lr',default=0.2,type=float,help="learning rate")
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=10,type=int,help="seed") 
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
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
            all_seed(env,seed= cfg['seed']) 
        if cfg['env_name'] == 'CliffWalking-v0':
            env = CliffWalkingWapper(env)
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        agent = Sarsa(cfg)
        return env,agent
        
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg['train_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0 # step per episode
            state = env.reset()  # reset and obtain initial state
            action = agent.sample_action(state)
            # while True:
            for _ in range(cfg['ep_max_steps']):
                next_state, reward, done, _ = env.step(action)  # update env and return transitions
                next_action =  agent.sample_action(next_state)
                agent.update(state, action, reward, next_state, next_action,done)  # update agent
                state = next_state  # update state
                action = next_action
                ep_reward += reward
                ep_step += 1
                if done:
                    break
            rewards.append(ep_reward)
            steps.append(ep_step)
            if (i_ep+1)%10==0:
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}, Steps: {ep_step}, Epislon: {agent.epsilon:.3f}')
        print("Finish training!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

    def test(self,cfg,env,agent):
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg['test_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg['ep_max_steps']):
                action = agent.predict_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                ep_reward+=reward
                ep_step+=1
                if done:
                    break  
            rewards.append(ep_reward)
            steps.append(ep_step)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}, Steps: {ep_step}, Reward: {ep_reward:.2f}")
        print("Finish testing!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
    
    

