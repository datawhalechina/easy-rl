#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 01:19:43
LastEditor: JiangJi
LastEditTime: 2022-11-01 01:21:06
Discription: 
'''
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
from common.utils import all_seed,merge_class_attrs
from common.launcher import Launcher
from common.memories import PGReplay
from common.models import ActorSoftmax,Critic
from envs.register import register_env
from a2c import A2C
from config.config import GeneralConfigA2C,AlgoConfigA2C

class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigA2C())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigA2C())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''  
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=True)  # create env
        if cfg.seed !=0: # set random seed
            all_seed(env,seed = cfg.seed) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        models = {'Actor':ActorSoftmax(n_states,n_actions, hidden_dim = cfg.actor_hidden_dim),'Critic':Critic(n_states,1,hidden_dim=cfg.critic_hidden_dim)}
        memories = {'ACMemory':PGReplay()}
        agent = A2C(models,memories,cfg)
        for k,v in models.items():
            logger.info(f"{k} model name: {type(v).__name__}")
        for k,v in memories.items():
            logger.info(f"{k} memory name: {type(v).__name__}")
        logger.info(f"agent name: {type(agent).__name__}")
        return env,agent
    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0 # step per episode
        ep_entropy = 0 # entropy per episode
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions
            agent.memory.push((agent.value,agent.log_prob,reward))  # save transitions
            state = next_state  # update state
            ep_reward += reward
            ep_entropy += agent.entropy
            ep_step += 1
            if terminated:
                break
        agent.update(next_state,ep_entropy)  # update agent
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0 # step per episode
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            action = agent.predict_action(state)  # predict action
            next_state, reward, terminated, truncated , info = env.step(action)  
            state = next_state 
            ep_reward += reward
            ep_step += 1
            if terminated:
                break
        return agent,ep_reward,ep_step
    # def train(self,cfg,env,agent,logger):
    #     logger.info("Start training!")
    #     logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    #     rewards = []  # record rewards for all episodes
    #     steps = [] # record steps for all episodes
    #     for i_ep in range(cfg.train_eps):
    #         ep_reward = 0  # reward per episode
    #         ep_step = 0 # step per episode
    #         ep_entropy = 0
    #         state = env.reset()  # reset and obtain initial state
    #         for _ in range(cfg.max_steps):
    #             action = agent.sample_action(state)  # sample action
    #             next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions
    #             agent.memory.push((agent.value,agent.log_prob,reward))  # save transitions
    #             state = next_state  # update state
    #             ep_reward += reward
    #             ep_entropy += agent.entropy
    #             ep_step += 1
    #             if terminated:
    #                 break
    #         agent.update(next_state,ep_entropy)  # update agent
    #         rewards.append(ep_reward)
    #         steps.append(ep_step)
    #         logger.info(f"Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.2f}, Steps:{ep_step}")
    #     logger.info("Finish training!")
    #     return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
    # def test(self,cfg,env,agent,logger):
    #     logger.info("Start testing!")
    #     logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    #     rewards = []  # record rewards for all episodes
    #     steps = [] # record steps for all episodes
    #     for i_ep in range(cfg.test_eps):
    #         ep_reward = 0  # reward per episode
    #         ep_step = 0
    #         state = env.reset()  # reset and obtain initial state
    #         for _ in range(cfg.max_steps):
    #             action = agent.predict_action(state)  # predict action
    #             next_state, reward, terminated, truncated , info = env.step(action)  
    #             state = next_state 
    #             ep_reward += reward
    #             ep_step += 1
    #             if terminated:
    #                 break
    #         rewards.append(ep_reward)
    #         steps.append(ep_step)
    #         logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.2f}, Steps:{ep_step}")
    #     logger.info("Finish testing!")
    #     env.close()
    #     return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
   

        
    
