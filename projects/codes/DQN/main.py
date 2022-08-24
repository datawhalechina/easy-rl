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
from common.utils import save_results,all_seed
from common.utils import plot_rewards,save_args
from common.models import MLP
from common.memories import ReplayBuffer
from dqn import DQN

def get_args():
    """ hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.95,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=500,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--target_update',default=4,type=int)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed") 
    parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
    # please manually change the following args in this script if you want
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models' )    
    args = parser.parse_args()    
    args = {**vars(args)}  # type(dict)         
    return args

def env_agent_config(cfg):
    ''' create env and agent
    '''
    env = gym.make(cfg['env_name'])  # create env
    if cfg['seed'] !=0: # set random seed
        all_seed(env,seed=cfg["seed"])
    n_states = env.observation_space.shape[0]  # state dimension
    n_actions = env.action_space.n  # action dimension
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
    model = MLP(n_states,n_actions,hidden_dim=cfg["hidden_dim"])
    memory =  ReplayBuffer(cfg["memory_capacity"]) # replay buffer
    agent = DQN(model,memory,cfg)  # create agent
    return env, agent

def train(cfg, env, agent):
    ''' 训练
    '''
    print("Start training!")
    print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
    rewards = []  # record rewards for all episodes
    steps = []
    for i_ep in range(cfg["train_eps"]):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        while True:
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, done, _ = env.step(action)  # update env and return transitions
            agent.memory.push(state, action, reward,
                              next_state, done)  # save transitions
            state = next_state  # update next state for env
            agent.update()  # update agent
            ep_reward += reward  #
            if done:
                break
        if (i_ep + 1) % cfg["target_update"] == 0:  # target net update, target_update means "C" in pseucodes
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}: Epislon: {agent.epsilon:.3f}')
    print("Finish training!")
    env.close()
    res_dic = {'episodes':range(len(rewards)),'rewards':rewards}
    return res_dic

def test(cfg, env, agent):
    print("Start testing!")
    print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
    rewards = []  # record rewards for all episodes
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        while True:
            ep_step+=1
            action = agent.predict_action(state)  # predict action
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
    return {'episodes':range(len(rewards)),'rewards':rewards}


if __name__ == "__main__":
    cfg = get_args()
    # training
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    save_args(cfg,path = cfg['result_path']) # save parameters
    agent.save_model(path = cfg['model_path'])  # save models
    save_results(res_dic, tag = 'train', path = cfg['result_path']) # save results
    plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "train")  # plot results
    # testing
    env, agent = env_agent_config(cfg) # create new env for testing, sometimes can ignore this step
    agent.load_model(path = cfg['model_path'])  # load model
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test',
                 path = cfg['result_path'])  
    plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "test")  
