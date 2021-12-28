import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import numpy as np
import datetime


from TD3.agent import TD3
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
		self.start_timestep = 25e3 # Time steps initial random policy is used
		self.eval_freq = 5e3 # How often (time steps) we evaluate
		# self.train_eps = 800
		self.max_timestep = 4000000 # Max time steps to run environment
		self.expl_noise = 0.1 # Std of Gaussian exploration noise
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.99 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.noise_clip = 0.5  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(env,agent, seed, eval_episodes=10):
	eval_env = gym.make(env)
	eval_env.seed(seed + 100)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			# eval_env.render()
			action = agent.choose_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def train(cfg,env,agent):
	# Evaluate untrained policy
	evaluations = [eval(cfg.env,agent, cfg.seed)]
	state, done = env.reset(), False
	ep_reward = 0
	ep_timesteps = 0
	episode_num = 0
	rewards = []
	ma_rewards = [] # moveing average reward
	for t in range(int(cfg.max_timestep)):
		ep_timesteps += 1
		# Select action randomly or according to policy
		if t < cfg.start_timestep:
			action = env.action_space.sample()
		else:
			action = (
				agent.choose_action(np.array(state))
				+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if ep_timesteps < env._max_episode_steps else 0
		# Store data in replay buffer
		agent.memory.push(state, action, next_state, reward, done_bool)
		state = next_state
		ep_reward += reward
		# Train agent after collecting sufficient data
		if t >= cfg.start_timestep:
			agent.update()
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
			evaluations.append(eval(cfg.env,agent, cfg.seed))
	return rewards, ma_rewards
# def train(cfg,env,agent):
# 	evaluations = [eval(cfg.env,agent,cfg.seed)]
# 	ep_reward = 0
# 	tot_timestep = 0
# 	rewards = []
# 	ma_rewards = [] # moveing average reward
# 	for i_ep in range(int(cfg.train_eps)):
# 		state, done = env.reset(), False
# 		ep_reward = 0
# 		ep_timestep = 0
# 		while not done:
# 			ep_timestep += 1
# 			tot_timestep +=1
# 			# Select action randomly or according to policy
# 			if tot_timestep < cfg.start_timestep:
# 				action = env.action_space.sample()
# 			else:
# 				action = (
# 					agent.choose_action(np.array(state))
# 					+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
# 				).clip(-max_action, max_action)
# 			# action = (
# 			# 		agent.choose_action(np.array(state))
# 			# 		+ np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
# 			# 	).clip(-max_action, max_action)
# 			# Perform action
# 			next_state, reward, done, _ = env.step(action) 
# 			done_bool = float(done) if ep_timestep < env._max_episode_steps else 0

# 			# Store data in replay buffer
# 			agent.memory.push(state, action, next_state, reward, done_bool)
# 			state = next_state
# 			ep_reward += reward
# 			# Train agent after collecting sufficient data
# 			if tot_timestep >= cfg.start_timestep:
# 				agent.update()
# 		print(f"Episode:{i_ep}/{cfg.train_eps}, Episode Timestep:{ep_timestep}, Reward:{ep_reward:.3f}")
# 		rewards.append(ep_reward)
# 		# 计算滑动窗口的reward
# 		if ma_rewards:
# 			ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
# 		else:
# 			ma_rewards.append(ep_reward) 
# 		# Evaluate episode
# 		if (i_ep+1) % cfg.eval_freq == 0:
# 			evaluations.append(eval(cfg.env,agent, cfg.seed))
# 	return rewards,ma_rewards


if __name__ == "__main__":
	cfg  = TD3Config()
	env = gym.make(cfg.env)
	env.seed(cfg.seed) # Set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	agent = TD3(state_dim,action_dim,max_action,cfg)
	rewards,ma_rewards = train(cfg,env,agent)
	make_dir(cfg.result_path,cfg.model_path)
	agent.save(path=cfg.model_path)
	save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
	plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
	# cfg.result_path = './TD3/results/HalfCheetah-v2/20210416-130341/'
	# agent.load(cfg.result_path)
	# eval(cfg.env,agent, cfg.seed)
	
		
