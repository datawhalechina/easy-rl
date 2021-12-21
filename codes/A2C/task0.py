import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import numpy as np
import torch
import torch.optim as optim
import datetime
from common.multiprocessing_env import SubprocVecEnv
from A2C.agent import ActorCritic
from common.utils import save_results, make_dir
from common.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
algo_name = 'A2C'  # 算法名称
env_name = 'CartPole-v0'  # 环境名称

class A2CConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name# 算法名称
        self.env_name = env_name # 环境名称
        self.n_envs = 8 # 异步的环境数目
        self.gamma = 0.99 # 强化学习中的折扣因子
        self.hidden_dim = 256
        self.lr = 1e-3 # learning rate
        self.max_frames = 30000
        self.n_steps = 5
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        

def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.seed(2)
        return env
    return _thunk
def test_env(env,model,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(cfg,envs):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo}, 设备：{cfg.device}')
    env = gym.make(cfg.env_name) # a single env
    env.seed(10)
    state_dim  = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    model = ActorCritic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    frame_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()
    while frame_idx < cfg.max_frames:
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        # rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            frame_idx += 1
            if frame_idx % 100 == 0:
                test_reward = np.mean([test_env(env,model) for _ in range(10)])
                print(f"frame_idx:{frame_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward) 
                # plot(frame_idx, test_rewards)   
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('完成训练！')
    return test_rewards, test_ma_rewards
if __name__ == "__main__":
    cfg = A2CConfig()
    plot_cfg = PlotConfig()
    envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs) 
    # 训练
    rewards,ma_rewards = train(cfg,envs)
    make_dir(plot_cfg.result_path,plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path) # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train") # 画出结果
