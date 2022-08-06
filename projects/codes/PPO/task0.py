import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import numpy as np
import datetime
import argparse
from common.utils import plot_rewards,save_args,save_results,make_dir
from ppo2 import PPO

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='PPO',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
    parser.add_argument('--continuous',default=False,type=bool,help="if PPO is continous") # PPO既可适用于连续动作空间，也可以适用于离散动作空间
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--batch_size',default=5,type=int) # mini-batch SGD中的批量大小
    parser.add_argument('--n_epochs',default=4,type=int)
    parser.add_argument('--actor_lr',default=0.0003,type=float,help="learning rate of actor net")
    parser.add_argument('--critic_lr',default=0.0003,type=float,help="learning rate of critic net")
    parser.add_argument('--gae_lambda',default=0.95,type=float)
    parser.add_argument('--policy_clip',default=0.2,type=float) # PPO-clip中的clip参数，一般是0.1~0.2左右
    parser.add_argument('--update_fre',default=20,type=int)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args
        
def env_agent_config(cfg,seed = 1):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    if cfg.continuous:
        n_actions = env.action_space.shape[0] # 动作维度
    else:
        n_actions = env.action_space.n  # 动作维度
    agent = PPO(n_states, n_actions, cfg)  # 创建智能体
    if seed !=0: # 设置随机种子
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent

def train(cfg,env,agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%10 == 0: 
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")
    print('完成训练！')
    env.close()
    res_dic = {'rewards':rewards,'ma_rewards':ma_rewards}
    return res_dic

def test(cfg,env,agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.test_eps, ep_reward))
    print('完成训练！')
    env.close()
    res_dic = {'rewards':rewards,'ma_rewards':ma_rewards}
    return res_dic

if __name__ == "__main__":
    cfg = get_args()
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  
    save_args(cfg) # 保存参数
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