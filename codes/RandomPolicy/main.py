#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-21 11:07:57
LastEditor: JiangJi
LastEditTime: 2021-04-21 11:15:00
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import numpy as np
import datetime

from common.plot import plot_rewards
from common.utils import save_results,make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
	

class TD3Config:
	def __init__(self) -> None:
		self.algo = 'TD3'
		self.env = 'HalfCheetah-v2'
		self.seed = 0
		self.result_path = curr_path+"/results/" +self.env+'/'+curr_time+'/results/'  # path to save results
		self.model_path = curr_path+"/results/" +self.env+'/'+curr_time+'/models/'  # path to save models
		self.eval_freq = 5e3 # How often (time steps) we evaluate
		# self.train_eps = 800
		self.max_timestep = 4000000 # Max time steps to run environment
		
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(env_name,seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			# eval_env.render()
			action = eval_env.action_space.sample()
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def train(cfg,env):
	# Evaluate untrained policy
	evaluations = [eval(cfg.env, cfg.seed)]
	state, done = env.reset(), False
	ep_reward = 0
	ep_timesteps = 0
	episode_num = 0
	rewards = []
	ma_rewards = [] # moveing average reward
	for t in range(int(cfg.max_timestep)):
		ep_timesteps += 1
		# Select action randomly
		action = env.action_space.sample()
		# Perform action
		next_state, reward, done, _ = env.step(action) 
		state = next_state
		ep_reward += reward
		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Episode:{episode_num+1}, Episode T:{ep_timesteps}, Reward:{ep_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			rewards.append(ep_reward)
			# 计算滑动窗口的reward
			if ma_rewards:
				ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
			else:
				ma_rewards.append(ep_reward) 
			ep_reward = 0
			ep_timesteps = 0
			episode_num += 1 
		# Evaluate episode
		if (t + 1) % cfg.eval_freq == 0:
			evaluations.append(eval(cfg.env, cfg.seed))
	return rewards, ma_rewards

if __name__ == "__main__":
	cfg  = TD3Config()
	env = gym.make(cfg.env)
	env.seed(cfg.seed) # Set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	rewards,ma_rewards = train(cfg,env)
	make_dir(cfg.result_path)
	save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
	plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
	# cfg.result_path = './TD3/results/HalfCheetah-v2/20210416-130341/'
	# agent.load(cfg.result_path)
	# eval(cfg.env,agent, cfg.seed)
	
		
