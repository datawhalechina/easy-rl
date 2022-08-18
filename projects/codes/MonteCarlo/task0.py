#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 14:26:44
LastEditor: John
LastEditTime: 2022-08-15 18:12:13
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import datetime
import argparse
from common.utils import save_results,save_args,plot_rewards

from MonteCarlo.agent import FisrtVisitMC
from envs.racetrack_env import RacetrackEnv

curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")  # obtain current time

def get_args():
    """ 超参数
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='First-Visit MC',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='Racetrack',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.9,type=float,help="discounted factor")
    parser.add_argument('--epsilon',default=0.15,type=float,help="the probability to select a random action")
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) 
    parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")           
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args

def env_agent_config(cfg,seed=1):
    env = RacetrackEnv()
    n_actions = env.action_space.n
    agent = FisrtVisitMC(n_actions, cfg)
    return env,agent
    
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
    cfg = get_args()
     # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    save_args(cfg,path = cfg.result_path) # 保存参数到模型路径上
    agent.save(path = cfg.model_path)  # 保存模型
    save_results(res_dic, tag = 'train', path = cfg.result_path)  
    plot_rewards(res_dic['rewards'], cfg, path = cfg.result_path,tag = "train")  
    # 测试
    env, agent = env_agent_config(cfg) # 也可以不加，加这一行的是为了避免训练之后环境可能会出现问题，因此新建一个环境用于测试
    agent.load(path = cfg.model_path)  # 导入模型
    res_dic = test(cfg, env, agent)
    save_results(res_dic, tag='test',
                 path = cfg.result_path)  # 保存结果
    plot_rewards(res_dic['rewards'], cfg, path = cfg.result_path,tag = "test")  # 画出结果
