import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path
import torch.nn as nn
import torch.nn.functional as F

import gym
import torch
import datetime
import numpy as np
import argparse
from common.utils import save_results, make_dir
from common.utils import plot_rewards,save_args
from dqn import DQN

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
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
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args

def env_agent_config(cfg,seed=1):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    print(f"n states: {n_states}, n actions: {n_actions}")
    agent = DQN(n_states,n_actions, cfg)  # 创建智能体
    if seed !=0: # 设置随机种子
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent

def train(cfg, env, agent):
    ''' Training
    '''
    print('Start training!')
    print(f'Env:{cfg.env_name}, A{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step += 1
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            print(f'Episode：{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} Epislon:{agent.epsilon(agent.frame_idx):.3f}')
    print('Finish training!')
    env.close()
    res_dic = {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}
    return res_dic


def test(cfg, env, agent):
    print('Start testing!')
    print(f'Env:{cfg.env_name}, A{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step+=1
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode：{i_ep+1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}')
    print('Finish testing')
    env.close()
    return {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}


if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  
    save_args(cfg) # save parameters
    agent.save(path=cfg.model_path)  # save model
    save_results(res_dic, tag='train',
                 path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'],cfg, tag="test")  # 画出结果
