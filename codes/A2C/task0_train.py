import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path


import gym
import numpy as np
import torch
import torch.optim as optim
import datetime
from common.multiprocessing_env import SubprocVecEnv
from A2C.model import ActorCritic
from common.utils import save_results, make_dir
from common.plot import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
class A2CConfig:
    def __init__(self) -> None:
        self.algo='A2C'
        self.env= 'CartPole-v0'
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.n_envs = 8
        self.gamma = 0.99
        self.hidden_size = 256
        self.lr = 1e-3 # learning rate
        self.max_frames = 30000
        self.n_steps = 5
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    env = gym.make(cfg.env) # a single env
    env.seed(10)
    state_dim  = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    model = ActorCritic(state_dim, action_dim, cfg.hidden_size).to(cfg.device)
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
    return test_rewards, test_ma_rewards
if __name__ == "__main__":
    cfg = A2CConfig()
    envs = [make_envs(cfg.env) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs) # 8 env
    rewards,ma_rewards = train(cfg,envs)
    make_dir(cfg.result_path,cfg.model_path)
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)
    plot_rewards(rewards,ma_rewards,tag="train",env=cfg.env,algo = cfg.algo,path=cfg.result_path)
