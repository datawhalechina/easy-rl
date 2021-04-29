

[Eng](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README_en.md)|[中文](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README.md)

## Introduction

This repo is used to learn basic RL algorithms, we will make it **detailed comment** and **clear structure** as much as possible:

The code structure mainly contains several scripts as following：

* ```model.py``` basic network model of RL, like MLP, CNN
* ```memory.py``` Replay Buffer
* ```plot.py``` use seaborn to plot rewards curve，saved in folder ``` result```.
* ```env.py``` to custom or normalize environments
* ```agent.py``` core algorithms, include a python Class with functions(choose action, update)
* ```main.py``` main function

Note that ```model.py```,```memory.py```,```plot.py``` shall be utilized in different algorithms，thus they are put into ```common``` folder。

## Runnig Environment

python 3.7、pytorch 1.6.0-1.7.1、gym 0.17.0-0.18.0
## Usage
运行带有```train```的py文件或ipynb文件进行训练，如果前面带有```task```如```task0_train.py```，表示对task0任务训练
类似的带有```eval```即为测试。

run python scripts or jupyter notebook file with ```train``` to train the agent, if there is a ```task``` like ```task0_train.py```, it means to train with task 0.

similar to file with ```eval```, which means to evaluate the agent.

## Schedule

|                   Name                   |                      Related materials                       | Used Envs                             | Notes |
| :--------------------------------------: | :----------------------------------------------------------: | ------------------------------------- | :---: |
| [On-Policy First-Visit MC](./MonteCarlo) |                                                              | [Racetrack](./envs/racetrack_env.md)  |       |
|        [Q-Learning](./QLearning)         |                                                              | [CliffWalking-v0](./envs/gym_info.md) |       |
|             [Sarsa](./Sarsa)             |                                                              | [Racetrack](./envs/racetrack_env.md)  |       |
|               [DQN](./DQN)               | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf),[Nature DQN Paper](https://www.nature.com/articles/nature14236) | [CartPole-v0](./envs/gym_info.md)     |       |
|           [DQN-cnn](./DQN_cnn)           | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  | [CartPole-v0](./envs/gym_info.md)     |       |
|         [DoubleDQN](./DoubleDQN)         |                                                              | [CartPole-v0](./envs/gym_info.md)     |       |
|   [Hierarchical DQN](HierarchicalDQN)    |     [Hierarchical DQN](https://arxiv.org/abs/1604.06057)     | [CartPole-v0](./envs/gym_info.md)     |       |
|    [PolicyGradient](./PolicyGradient)    |                                                              | [CartPole-v0](./envs/gym_info.md)     |       |
|               [A2C](./A2C)               |        [A3C Paper](https://arxiv.org/abs/1602.01783)         | [CartPole-v0](./envs/gym_info.md)     |       |
|               [SAC](./SAC)               |        [SAC Paper](https://arxiv.org/abs/1801.01290)         |                                       |       |
|               [PPO](./PPO)               |        [PPO paper](https://arxiv.org/abs/1707.06347)         | [CartPole-v0](./envs/gym_info.md)     |       |
|              [DDPG](./DDPG)              |        [DDPG Paper](https://arxiv.org/abs/1509.02971)        | [Pendulum-v0](./envs/gym_info.md)     |       |
|               [TD3](./TD3)               |        [TD3 Paper](https://arxiv.org/abs/1802.09477)         | HalfCheetah-v2                        |       |

## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)
