import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import numpy as np

def train(cfg, env, agent):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            goal = agent.set_goal(state)
            onehot_goal = agent.to_onehot(goal)
            meta_state = state
            extrinsic_reward = 0
            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onehot_goal])
                action = agent.choose_action(goal_state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == np.argmax(
                    next_state) else 0.0
                agent.memory.push(goal_state, action, intrinsic_reward, np.concatenate(
                    [next_state, onehot_goal]), done)
                state = next_state
                agent.update()
        if (i_ep+1)%10 == 0: 
            print(f'回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward}，Loss:{agent.loss_numpy:.2f}， Meta_Loss:{agent.meta_loss_numpy:.2f}')
        agent.meta_memory.push(meta_state, goal, extrinsic_reward, state, done)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards

def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            goal = agent.set_goal(state)
            onehot_goal = agent.to_onehot(goal)
            extrinsic_reward = 0
            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onehot_goal])
                action = agent.choose_action(goal_state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                extrinsic_reward += reward
                state = next_state
                agent.update()
        if (i_ep+1)%10 == 0: 
            print(f'回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward}，Loss:{agent.loss_numpy:.2f}， Meta_Loss:{agent.meta_loss_numpy:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards