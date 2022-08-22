import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import argparse
import datetime
import gym
import torch
import random
import numpy as np
import torch.nn as nn
from common.memories import ReplayBufferQue
from common.models import MLP
from common.utils import save_results,all_seed,plot_rewards,save_args
from softq import SoftQ

def get_args():
    """ hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='SoftQ',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--max_steps',default=200,type=int,help="maximum steps per episode")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--alpha',default=4,type=float,help="alpha")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=50000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) 
    parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args

class SoftQNetwork(nn.Module):
    '''Actually almost same to common.models.MLP
    '''
    def __init__(self,input_dim,output_dim):
        super(SoftQNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def env_agent_config(cfg):
    ''' create env and agent
    '''
    env = gym.make(cfg.env_name)  # create env
    if cfg.seed !=0: # set random seed
        all_seed(env,seed=cfg.seed)
    n_states = env.observation_space.shape[0]  # state dimension
    n_actions = env.action_space.n  # action dimension
    print(f"state dim: {n_states}, action dim: {n_actions}")
    # model = MLP(n_states,n_actions)
    model = SoftQNetwork(n_states,n_actions)
    memory =  ReplayBufferQue(cfg.memory_capacity) # replay buffer
    agent = SoftQ(n_actions,model,memory,cfg)  # create agent
    return env, agent

def train(cfg, env, agent):
    ''' training
    '''
    print("start training!")
    print(f"Env: {cfg.env_name}, Algo: {cfg.algo_name}, Device: {cfg.device}")
    rewards = []  # record rewards for all episodes
    steps = [] # record steps for all episodes, sometimes need
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()   # reset and obtain initial state
        while True:
        # for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state) # sample action
            next_state, reward, done, _ = env.step(action)  # update env and return transitions
            agent.memory.push((state, action, reward, next_state, done))  # save transitions
            state = next_state  # update next state for env
            agent.update()  # update agent
            ep_reward += reward  
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # target net update, target_update means "C" in pseucodes
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f'Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.2f}')
    print("finish training!")
    res_dic = {'episodes':range(len(rewards)),'rewards':rewards}
    return res_dic
def test(cfg, env, agent):
    print("start testing!")
    print(f"Env: {cfg.env_name}, Algo: {cfg.algo_name}, Device: {cfg.device}")
    rewards = []  # record rewards for all episodes
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # reward per episode
        state = env.reset()  # reset and obtain initial state
        while True:
            action = agent.predict_action(state)  # predict action
            next_state, reward, done, _ = env.step(action)  
            state = next_state  
            ep_reward += reward 
            if done:
                break
        rewards.append(ep_reward)
        print(f'Episode: {i_ep+1}/{cfg.test_eps}，Reward: {ep_reward:.2f}')
    print("finish testing!")
    env.close()
    return {'episodes':range(len(rewards)),'rewards':rewards}

if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    save_args(cfg,path = cfg.result_path) # 保存参数到模型路径上
    agent.save_model(path = cfg.model_path)  # 保存模型
    save_results(res_dic, tag = 'train', path = cfg.result_path)  
    plot_rewards(res_dic['rewards'], cfg, path = cfg.result_path,tag = "train")  
    # 测试
    env, agent = env_agent_config(cfg) # 也可以不加，加这一行的是为了避免训练之后环境可能会出现问题，因此新建一个环境用于测试
    agent.load_model(path = cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test',
                 path = cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], cfg, path = cfg.result_path,tag = "test")  # 画出结果