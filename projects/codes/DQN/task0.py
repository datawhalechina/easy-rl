import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime
import numpy as np
import argparse
from common.utils import save_results
from common.utils import plot_rewards,save_args
from common.models import MLP
from common.memories import ReplayBuffer
from dqn import DQN

def get_args():
    """ 超参数
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
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
            '/' + curr_time + '/models/' ) 
    parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()                          
    return args

def env_agent_config(cfg,seed=1):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    print(f"状态数：{n_states}，动作数：{n_actions}")
    model = MLP(n_states,n_actions,hidden_dim=cfg.hidden_dim)
    memory =  ReplayBuffer(cfg.memory_capacity) # 经验回放
    agent = DQN(n_actions,model,memory,cfg)  # 创建智能体
    if seed !=0: # 设置随机种子
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent

def train(cfg, env, agent):
    ''' 训练
    '''
    print("开始训练！")
    print(f"回合：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step += 1
            action = agent.sample(state)  # 选择动作
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
        if (i_ep + 1) % 10 == 0:
            print(f'回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}')
    print("完成训练！")
    env.close()
    res_dic = {'rewards':rewards}
    return res_dic

def test(cfg, env, agent):
    print("开始测试！")
    print(f"回合：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            ep_step+=1
            action = agent.predict(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}')
    print("完成测试")
    env.close()
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
