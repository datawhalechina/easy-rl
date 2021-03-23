
[Eng](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README_en.md)|[中文](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README.md)

## 写在前面

本项目用于学习RL基础算法，尽量做到: **注释详细**，**结构清晰**。

代码结构主要分为以下几个脚本：

* ```model.py``` 强化学习算法的基本模型，比如神经网络，actor，critic等
* ```memory.py``` 保存Replay Buffer，用于off-policy
* ```plot.py``` 利用matplotlib或seaborn绘制rewards图，包括滑动平均的reward，结果保存在result文件夹中
* ```env.py``` 用于构建强化学习环境，也可以重新自定义环境，比如给action加noise
* ```agent.py``` RL核心算法，比如dqn等，主要包含update和choose_action两个方法，
* ```main.py``` 运行主函数

其中```model.py```,```memory.py```,```plot.py``` 由于不同算法都会用到，所以放入```common```文件夹中。

## 运行环境

python 3.7、pytorch 1.6.0-1.7.1、gym 0.17.0-0.18.0
## 使用说明

对应算法文件夹下运行```main.py```即可
## 算法进度

|                 算法名称                 |                        相关论文材料                         | 环境                                  |                备注                |
| :--------------------------------------: | :---------------------------------------------------------: | ------------------------------------- | :--------------------------------: |
| [On-Policy First-Visit MC](./MonteCarlo) |                                                             | [Racetrack](./envs/racetrack_env.md)  |                                    |
|        [Q-Learning](./QLearning)         |                                                             | [CliffWalking-v0](./envs/gym_info.md) |                                    |
|             [Sarsa](./Sarsa)             |                                                             | [Racetrack](./envs/racetrack_env.md)  |                                    |
|               [DQN](./DQN)               | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [CartPole-v0](./envs/gym_info.md)     |                                    |
|                 DQN-cnn                  | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [CartPole-v0](./envs/gym_info.md)     | 与DQN相比使用了CNN而不是全链接网络 |
|         [DoubleDQN](./DoubleDQN)         |                                                             | [CartPole-v0](./envs/gym_info.md)     |          效果不好，待改进          |
|             Hierarchical DQN             |    [Hierarchical DQN](https://arxiv.org/abs/1604.06057)     |                                       |                                    |
|    [PolicyGradient](./PolicyGradient)    |                                                             | [CartPole-v0](./envs/gym_info.md)     |                                    |
|                   A2C                    |                                                             | [CartPole-v0](./envs/gym_info.md)     |                                    |
|                   A3C                    |                                                             |                                       |                                    |
|                   SAC                    |                                                             |                                       |                                    |
|               [PPO](./PPO)               |        [PPO paper](https://arxiv.org/abs/1707.06347)        | [CartPole-v0](./envs/gym_info.md)     |                                    |
|                   DDPG                   |       [DDPG Paper](https://arxiv.org/abs/1509.02971)        | [Pendulum-v0](./envs/gym_info.md)     |                                    |
|                   TD3                    | [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477) |                                       |                                    |
|                   GAIL                   |                                                             |                                       |                                    |




## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)

https://www.cnblogs.com/lucifer1997/p/13458563.html
