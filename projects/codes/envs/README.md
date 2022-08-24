# 环境说明汇总

## 算法SAR一览

说明：SAR分别指状态(S)、动作(A)以及奖励(R)，下表的Reward Range表示每回合能获得的奖励范围，Steps表示环境中每回合的最大步数

|           Environment ID           | Observation Space | Action Space | Reward Range |  Steps   |
| :--------------------------------: | :---------------: | :----------: | :----------: | :------: |
|            CartPole-v0             |      Box(4,)      | Discrete(2)  |   [0,200]    |   200    |
|            CartPole-v1             |      Box(4,)      | Discrete(2)  |   [0,500]    |   500    |
|          CliffWalking-v0           |   Discrete(48)    | Discrete(4)  |  [-inf,-13]  | [13,inf] |
| FrozenLake-v1(*is_slippery*=False) |   Discrete(16)    | Discrete(4)  |    0 or 1    | [6,info] |

## 环境描述

[OpenAI Gym](./gym_info.md)  
[MuJoCo](./mujoco_info.md)  

