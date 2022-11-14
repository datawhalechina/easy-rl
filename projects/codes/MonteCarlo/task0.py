#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 14:26:44
LastEditor: John
LastEditTime: 2022-11-08 23:35:18
Discription: 
Environment: 
'''
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import datetime
import gym
from envs.wrappers import CliffWalkingWapper
from envs.register import register_env
from common.utils import merge_class_attrs,all_seed
from common.launcher import Launcher
from MonteCarlo.agent import FisrtVisitMC
from MonteCarlo.config.config import GeneralConfigMC,AlgoConfigMC

class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigMC())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigMC())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''  
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=False)  # create env
        if cfg.env_name == 'CliffWalking-v0':
            env = CliffWalkingWapper(env)
        if cfg.seed !=0: # set random seed
            all_seed(env,seed=cfg.seed) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        agent = FisrtVisitMC(cfg)
        return env,agent
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        one_ep_transition = []
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            one_ep_transition.append((state, action, reward))  # save transitions
            agent.update(one_ep_transition)  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
    
def train(cfg, env, agent):
    print("开始训练！")
    print(f"环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}")
    rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        one_ep_transition = []
        while True:
            action = agent.sample(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            one_ep_transition.append((state, action, reward))
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        agent.update(one_ep_transition)
        if (i_ep+1) % 10 == 0:
            print(f"Episode:{i_ep+1}/{cfg.train_eps}: Reward:{ep_reward}")
    print("完成训练")
    return {'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    print(f"环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}")
    rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.predict(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f'回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}')
    return {'rewards':rewards}
    
if __name__ == "__main__":
    main = Main()
    main.run()