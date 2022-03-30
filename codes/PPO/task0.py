import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import gym
import torch
import datetime
from common.plot import plot_rewards
from common.utils import save_results,make_dir
from PPO.agent import PPO
from PPO.train import train

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class PPOConfig:
    def __init__(self) -> None:
        self.algo = "DQN"  # 算法名称
        self.env_name = 'CartPole-v0' # 环境名称
        self.continuous = False # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        self.batch_size = 5
        self.gamma=0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda=0.95
        self.policy_clip=0.2
        self.hidden_dim = 256
        self.update_fre = 20 # frequency of agent update

class PlotConfig:
    def __init__(self) -> None:
        self.algo = "DQN"  # 算法名称
        self.env_name = 'CartPole-v0' # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片

def env_agent_config(cfg,seed=1):
    env = gym.make(cfg.env_name)  
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim,action_dim,cfg)
    return env,agent

cfg  = PPOConfig()
plot_cfg = PlotConfig()
# 训练
env,agent = env_agent_config(cfg,seed=1)
rewards, ma_rewards = train(cfg, env, agent)
make_dir(plot_cfg.result_path, plot_cfg.model_path) # 创建保存结果和模型路径的文件夹
agent.save(path=plot_cfg.model_path)
save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")
# 测试
env,agent = env_agent_config(cfg,seed=10)
agent.load(path=plot_cfg.model_path)
rewards,ma_rewards = eval(cfg,env,agent)
save_results(rewards,ma_rewards,tag='eval',path=plot_cfg.result_path)
plot_rewards(rewards,ma_rewards,plot_cfg,tag="eval")