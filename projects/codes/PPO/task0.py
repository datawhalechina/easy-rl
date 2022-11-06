import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
import torch
import datetime
import numpy as np
import argparse
import torch.nn as nn


from common.utils import all_seed,merge_class_attrs
from common.models import ActorSoftmax, Critic
from common.memories import PGReplay
from common.launcher import Launcher
from envs.register import register_env
from ppo2 import PPO
from config,config import GeneralConfigPPO,AlgoConfigPPO
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.terminateds = []
        self.batch_size = batch_size
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),np.array(self.actions),np.array(self.probs),\
                np.array(self.vals),np.array(self.rewards),np.array(self.terminateds),batches
                
    def push(self, state, action, probs, vals, reward, terminated):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.terminateds.append(terminated)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.terminateds = []
        self.vals = []


class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigPPO())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigPPO())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=False)  # create env
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
        models = {'Actor':ActorSoftmax(n_states,n_actions, hidden_dim = cfg.actor_hidden_dim),'Critic':Critic(n_states,1,hidden_dim=cfg.critic_hidden_dim)}
        memory =  PGReplay # replay buffer
        agent = PPO(models,memory,cfg)  # create agent
        return env, agent
    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0 # step per episode
        state = env.reset()
        for _ in range(cfg.max_steps):
            action, prob, val = agent.sample_action(state)
            next_state, reward, terminated, _ = env.step(action)
            ep_reward += reward
            ep_step += 1
            agent.memory.push((state, action, prob, val, reward, terminated))
            if ep_step % cfg['update_fre'] == 0:
                agent.update()
            state = next_state
            if terminated:
                break
        return agent, ep_reward, ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0 # step per episode
        state = env.reset()
        for _ in range(cfg.max_steps):
            action, prob, val = agent.sample_action(state)
            next_state, reward, terminated, _ = env.step(action)
            ep_reward += reward
            ep_step += 1
            state = next_state
            if terminated:
                break
        return agent, ep_reward, ep_step
    def train(self,cfg,env,agent):
        ''' train agent
        '''
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = 0
        for i_ep in range(cfg['train_eps']):
            state = env.reset()
            ep_reward = 0
            while True:
                action, prob, val = agent.sample_action(state)
                next_state, reward, terminated, _ = env.step(action)
                steps += 1
                ep_reward += reward
                agent.memory.push(state, action, prob, val, reward, terminated)
                if steps % cfg['update_fre'] == 0:
                    agent.update()
                state = next_state
                if terminated:
                    break
            rewards.append(ep_reward)
            if (i_ep+1)%10==0:
                print(f"Episode: {i_ep+1}/{cfg['train_eps']}, Reward: {ep_reward:.2f}")
        print("Finish training!")
        return {'episodes':range(len(rewards)),'rewards':rewards}
    def test(self,cfg,env,agent):
        ''' test agent
        '''
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        for i_ep in range(cfg['test_eps']):
            state = env.reset()
            ep_reward = 0
            while True:
                action, prob, val = agent.predict_action(state)
                next_state, reward, terminated, _ = env.step(action)
                ep_reward += reward
                state = next_state
                if terminated:
                    break
            rewards.append(ep_reward)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}, Reward: {ep_reward:.2f}")
        print("Finish testing!")
        return {'episodes':range(len(rewards)),'rewards':rewards}

if __name__ == "__main__":
    main = Main()
    main.run()