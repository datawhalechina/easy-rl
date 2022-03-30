# 使用Policy-Based方法实现Pendulum-v0

使用Policy-Based方法比如DDPG等实现Pendulum-v0环境

## Pendulum-v0

![image-20200820174814084](img/image-20200820174814084.png)

钟摆以随机位置开始，目标是将其摆动，使其保持向上直立。动作空间是连续的，值的区间为[-2,2]。每个step给的reward最低为-16.27，最高为0。

环境建立如下：

```python
env = gym.make('Pendulum-v0') 
env.seed(1) # 设置env随机种子
n_states = env.observation_space.shape[0] # 获取总的状态数
```

## 强化学习基本接口

```python
rewards = [] # 记录总的rewards
moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
ep_steps = []
for i_episode in range(1, cfg.max_episodes+1): # cfg.max_episodes为最大训练的episode数
    state = env.reset() # reset环境状态
    ep_reward = 0
    for i_step in range(1, cfg.max_steps+1): # cfg.max_steps为每个episode的补偿
        action = agent.select_action(state) # 根据当前环境state选择action
        next_state, reward, done, _ = env.step(action) # 更新环境参数
        ep_reward += reward
        agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
        state = next_state # 跳转到下一个状态
        agent.update() # 每步更新网络
        if done:
            break
    # 更新target network，复制DQN中的所有weights and biases
    if i_episode % cfg.target_update == 0: #  cfg.target_update为target_net的更新频率
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print('Episode:', i_episode, ' Reward: %i' %
          int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
    ep_steps.append(i_step)
    rewards.append(ep_reward)
    # 计算滑动窗口的reward
    if i_episode == 1:
        moving_average_rewards.append(ep_reward)
    else:
        moving_average_rewards.append(
            0.9*moving_average_rewards[-1]+0.1*ep_reward)
```

## 任务要求

训练并绘制reward以及滑动平均后的reward随episode的变化曲线图并记录超参数写成报告，图示如下：

![rewards_train](assets/rewards_train.png)

![moving_average_rewards_train](assets/moving_average_rewards_train.png)

![steps_train](assets/steps_train.png)

同时也可以绘制测试(eval)模型时的曲线：

![rewards_eval](assets/rewards_eval.png)

![moving_average_rewards_eval](assets/moving_average_rewards_eval.png)

![steps_eval](assets/steps_eval.png)

也可以[tensorboard](https://pytorch.org/docs/stable/tensorboard.html)查看结果，如下：

![image-20201015221602396](assets/image-20201015221602396.png)

### 注意

1. 本次环境action范围在[-2,2]之间，而神经网络中输出的激活函数tanh在[0,1]，可以使用NormalizedActions(gym.ActionWrapper)的方法解决
2. 由于本次环境为惯性系统，建议增加Ornstein-Uhlenbeck噪声提高探索率，可参考[知乎文章](https://zhuanlan.zhihu.com/p/96720878)
3. 推荐多次试验保存rewards，然后使用searborn绘制，可参考[CSDN](https://blog.csdn.net/JohnJim0/article/details/106715402)

### 代码清单

**main.py**：保存强化学习基本接口，以及相应的超参数，可使用argparse

**model.py**：保存神经网络，比如全链接网络

**ddpg.py**: 保存算法模型，主要包含select_action和update两个函数

**memory.py**：保存Replay Buffer

**plot.py**：保存相关绘制函数

**noise.py**：保存噪声相关

[参考代码](https://github.com/datawhalechina/easy-rl/tree/master/codes/DDPG)

