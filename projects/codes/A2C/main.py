import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import datetime
import argparse
import gym
import torch
import numpy as np
from common.utils import all_seed
from common.launcher import Launcher
from common.memories import PGReplay
from common.models import ActorSoftmax,Critic
from envs.register import register_env
from a2c import A2C

class Main(Launcher):
    def get_args(self):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='A2C',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=1600,type=int,help="episodes of training") 
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing") 
        parser.add_argument('--ep_max_steps',default = 100000,type=int,help="steps per episode, much larger value can simulate infinite steps")
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor") 
        parser.add_argument('--actor_lr',default=3e-4,type=float,help="learning rate of actor")
        parser.add_argument('--critic_lr',default=1e-3,type=float,help="learning rate of critic")
        parser.add_argument('--actor_hidden_dim',default=256,type=int,help="hidden of actor net")
        parser.add_argument('--critic_hidden_dim',default=256,type=int,help="hidden of critic net")
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
        models = {'Actor':ActorSoftmax(cfg['n_states'],cfg['n_actions'], hidden_dim = cfg['actor_hidden_dim']),'Critic':Critic(cfg['n_states'],1,hidden_dim=cfg['critic_hidden_dim'])}
        memories = {'ACMemory':PGReplay()}
        agent = A2C(models,memories,cfg)
        return env,agent
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        
        for i_ep in range(cfg['train_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0 # step per episode
            ep_entropy = 0
            state = env.reset()  # reset and obtain initial state
            
            for _ in range(cfg['ep_max_steps']):
                action, value, dist = agent.sample_action(state)  # sample action
                next_state, reward, done, _ = env.step(action)  # update env and return transitions
                log_prob = torch.log(dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist.detach().numpy()) * np.log(dist.detach().numpy()))
                agent.memory.push((value,log_prob,reward))  # save transitions
                state = next_state  # update state
                ep_reward += reward
                ep_entropy += entropy
                ep_step += 1
                if done:
                    break
            agent.update(next_state,ep_entropy)  # update agent
            rewards.append(ep_reward)
            steps.append(ep_step)
            if (i_ep+1)%10==0:
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}, Steps:{ep_step}')
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
                action,_,_ = agent.predict_action(state)  # predict action
                next_state, reward, done, _ = env.step(action)  
                state = next_state 
                ep_reward += reward
                ep_step += 1
                if done:
                    break
            rewards.append(ep_reward)
            steps.append(ep_step)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}, Steps:{ep_step}, Reward: {ep_reward:.2f}")
        print("Finish testing!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
   

        
    
