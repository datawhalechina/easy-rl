import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import numpy as np
import datetime
from common.utils import plot_rewards
from common.utils import save_results,make_dir
from ppo2 import PPO

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class Config:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "DQN"  # 算法名称
        self.env_name = 'CartPole-v0' # 环境名称
        self.continuous = False # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10 # 随机种子，置0则不设置随机种子
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.batch_size = 5  # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.n_epochs = 4
        self.actor_lr = 0.0003 # actor的学习率
        self.critic_lr = 0.0003 # critic的学习率
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20 # 策略更新频率
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        ################################################################################
        
def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    if cfg.continuous:
        n_actions = env.action_space.shape[0] # 动作维度
    else:
        n_actions = env.action_space.n  # 动作维度
    agent = PPO(n_states, n_actions, cfg)  # 创建智能体
    if cfg.seed !=0: # 设置随机种子
        torch.manual_seed(cfg.seed)
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
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
    return rewards,ma_rewards

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
    return rewards,ma_rewards

if __name__ == "__main__":
    cfg  = Config()
    # 训练
    env,agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path) # 创建保存结果和模型路径的文件夹
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")
    # 测试
    env,agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = test(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='test',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,cfg,tag="test")