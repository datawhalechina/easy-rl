

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
run python scripts or jupyter notebook file with ```train``` to train the agent, if there is a ```task``` like ```task0_train.py```, it means to train with task 0.

similar to file with ```eval```, which means to evaluate the agent.

## Schedule

|                   Name                   |                      Related materials                       | Used Envs                                 | Notes |
| :--------------------------------------: | :----------------------------------------------------------: | ----------------------------------------- | :---: |
| [On-Policy First-Visit MC](./MonteCarlo) | [medium blog](https://medium.com/analytics-vidhya/monte-carlo-methods-in-reinforcement-learning-part-1-on-policy-methods-1f004d59686a) | [Racetrack](./envs/racetrack_env.md)      |                                    |
|        [Q-Learning](./QLearning)         | [towardsdatascience blog](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56),[q learning paper](https://ieeexplore.ieee.org/document/8836506) | [CliffWalking-v0](./envs/gym_info.md)     |                                    |
|             [Sarsa](./Sarsa)             | [geeksforgeeks blog](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/) | [Racetrack](./envs/racetrack_env.md)      |                                    |
|               [DQN](./DQN)               | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf),[Nature DQN Paper](https://www.nature.com/articles/nature14236) | [CartPole-v0](./envs/gym_info.md)         |                                    |
|           [DQN-cnn](./DQN_cnn)           | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  | [CartPole-v0](./envs/gym_info.md)         |  |
|         [DoubleDQN](./DoubleDQN)         |     [DoubleDQN Paper](https://arxiv.org/abs/1509.06461)      | [CartPole-v0](./envs/gym_info.md)         |                                    |
|   [Hierarchical DQN](HierarchicalDQN)    |       [H-DQN Paper](https://arxiv.org/abs/1604.06057)        | [CartPole-v0](./envs/gym_info.md)         |                                    |
|    [PolicyGradient](./PolicyGradient)    | [Lil'log](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) | [CartPole-v0](./envs/gym_info.md)         |                                    |
|               [A2C](./A2C)               |        [A3C Paper](https://arxiv.org/abs/1602.01783)         | [CartPole-v0](./envs/gym_info.md)         |                                    |
|               [SAC](./SAC)               |        [SAC Paper](https://arxiv.org/abs/1801.01290)         | [Pendulum-v0](./envs/gym_info.md)         |                                    |
|               [PPO](./PPO)               |        [PPO paper](https://arxiv.org/abs/1707.06347)         | [CartPole-v0](./envs/gym_info.md)         |                                    |
|              [DDPG](./DDPG)              |        [DDPG Paper](https://arxiv.org/abs/1509.02971)        | [Pendulum-v0](./envs/gym_info.md)         |                                    |
|               [TD3](./TD3)               |        [TD3 Paper](https://arxiv.org/abs/1802.09477)         | [HalfCheetah-v2]((./envs/mujoco_info.md)) |                                    |


## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)