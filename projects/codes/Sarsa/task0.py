#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2022-08-04 22:28:51
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import datetime
import argparse
from envs.racetrack_env import RacetrackEnv
from Sarsa.sarsa import Sarsa
from common.utils import save_results,make_dir,plot_rewards,save_args

def get_args():
    """ 超参数
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='Sarsa',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CliffWalking-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=300,type=int,help="episodes of training") # 训练的回合数
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing") # 测试的回合数
    parser.add_argument('--ep_max_steps',default=200,type=int) # 每回合最大的部署
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor") # 折扣因子
    parser.add_argument('--epsilon_start',default=0.90,type=float,help="initial value of epsilon") #  e-greedy策略中初始epsilon
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon") # e-greedy策略中的终止epsilon
    parser.add_argument('--epsilon_decay',default=200,type=int,help="decay rate of epsilon") # e-greedy策略中epsilon的衰减率
    parser.add_argument('--lr',default=0.2,type=float,help="learning rate")
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args


def env_agent_config(cfg,seed=1):
    env = RacetrackEnv()
    n_actions = 9 # 动作数
    agent = Sarsa(n_actions,cfg)
    return env,agent
        
def train(cfg,env,agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        action = agent.sample(state)
        ep_reward = 0
        # while True:
        for _ in range(cfg.ep_max_steps):
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            next_action = agent.sample(next_state)
            agent.update(state, action, reward, next_state, next_action,done)
            state = next_state
            action = next_action
            if done:
                break  
        rewards.append(ep_reward)
        if (i_ep+1)%2==0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon}")
    print('完成训练！')
    return {"rewards":rewards}

def test(cfg,env,agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        # while True:
        for _ in range(cfg.ep_max_steps):
            action = agent.predict(state)
            next_state, reward, done = env.step(action)
            ep_reward+=reward
            state = next_state
            if done:
                break  
        rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards":rewards}
        
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
    plot_rewards(res_dic['rewards'], cfg, tag="train")  
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
    
    

