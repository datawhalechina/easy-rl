前面项目讲的环境都是离散动作的，但实际中也有很多连续动作的环境，比如Open AI Gym中的[Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0)环境，它解决的是一个倒立摆问题，我们先对该环境做一个简要说明。

## Pendulum-v0简介

如果说 CartPole-v0 是一个离散动作的经典入门环境的话，那么对应 Pendulum-v0 就是连续动作的经典入门环境，如下图，我们通过施加力矩使其向上摆动并保持直立。

<img src="../../easy_rl_book/res/ch12/assets/pendulum_1.png" alt="image-20210915161550713" style="zoom:50%;" />

该环境的状态维度有三个，设摆针竖直方向上的顺时针旋转角为$\theta$，$\theta$设在$[-\pi,\pi]$之间，则相应的状态为$[cos\theta,sin\theta,\dot{\theta}]$，即表示角度和角速度，我们的动作则是一个-2到2之间的力矩，它是一个连续量，因而该环境不能用离散动作的算法比如 DQN 来解决。关于奖励是根据相关的物理原理而计算出的等式，如下：
$$
-\left(\theta^{2}+0.1 * \hat{\theta}^{2}+0.001 * \text { action }^{2}\right)
$$
对于每一步，其最低奖励为$-\left(\pi^{2}+0.1 * 8^{2}+0.001 *  2^{2}\right)= -16.2736044$，最高奖励为0。同 CartPole-v0 环境一样，达到最优算法的情况下，每回合的步数是无限的，因此这里设定每回合最大步数为200以便于训练。

##  DDPG 基本接口

我们依然使用接口的概念，通过伪代码分析并实现 DDPG 的训练模式，如下：

> 初始化评论家网络$Q\left(s, a \mid \theta^{Q}\right)$和演员网络$\mu\left(s \mid \theta^{\mu}\right)$，其权重分别为$\theta^{Q}$和$\theta^{\mu}$
>
> 初始化目标网络$Q'$和$\mu'$，并复制权重$\theta^{Q^{\prime}} \leftarrow \theta^{Q}, \theta^{\mu^{\prime}} \leftarrow \theta^{\mu}$
>
> 初始化经验回放缓冲区$R$
>
> 执行$M$个回合循环，对于每个回合：
>
> * 初始化动作探索的的随机过程即噪声$\mathcal{N}$
>
> * 初始化状态$s_1$
>
>   循环$T$个时间步长，对于每个时步$
>
>   * 根据当前策略和噪声选择动作$a_{t}=\mu\left(s_{t} \mid \theta^{\mu}\right)+\mathcal{N}_{t}$
>   * 执行动作$a_t$并得到反馈$r_t$和下一个状态$s_{t+1}$
>   * 存储转移$\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$到经验缓冲$R$中
>   * (更新策略)从$D$随机采样一个小批量的转移
>   * (更新策略)计算实际的Q值$y_{i}=r_{i}+\gamma Q^{\prime}\left(s_{i+1}, \mu^{\prime}\left(s_{i+1} \mid \theta^{\mu^{\prime}}\right) \mid \theta^{Q^{\prime}}\right)$
>   * (更新策略)对损失函数$L=\frac{1}{N} \sum_{i}\left(y_{i}-Q\left(s_{i}, a_{i} \mid \theta^{Q}\right)\right)^{2}$关于参数$\theta$做梯度下降用于更新评论家网络
>   * (更新策略)使用采样梯度更新演员网络的策略：$\left.\left.\nabla_{\theta^{\mu}} J \approx \frac{1}{N} \sum_{i} \nabla_{a} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{i}, a=\mu\left(s_{i}\right)} \nabla_{\theta^{\mu}} \mu\left(s \mid \theta^{\mu}\right)\right|_{s_{i}}$
>   * (更新策略)更新目标网络：$\theta^{Q^{\prime}} \leftarrow \tau \theta^{Q}+(1-\tau) \theta^{Q^{\prime}}$，$\theta^{\mu^{\prime}} \leftarrow \tau \theta^{\mu}+(1-\tau) \theta^{\mu^{\prime}}$

代码如下：

```python
ou_noise = OUNoise(env.action_space)  # 动作噪声
rewards = [] # 记录奖励
ma_rewards = []  # 记录滑动平均奖励
for i_ep in range(cfg.train_eps):
    state = env.reset()
    ou_noise.reset()
    done = False
    ep_reward = 0
    i_step = 0
    while not done:
        i_step += 1
        action = agent.choose_action(state)
        action = ou_noise.get_action(action, i_step) 
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        agent.memory.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
    if (i_ep+1)%10 == 0:
        print('回合：{}/{}，奖励：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
    else:
        ma_rewards.append(ep_reward)
```

相比于 DQN ，DDPG 主要多了两处修改，一个是给动作施加噪声，另外一个是软更新策略，即最后一步。

## Ornstein-Uhlenbeck噪声

 OU 噪声适用于惯性系统，尤其是时间离散化粒度较小的情况。 OU 噪声是一种随机过程，下面略去证明，直接给出公式：
$$
x(t+\Delta t)=x(t)-\theta(x(t)-\mu) \Delta t+\sigma W_t
$$
其中 $W_t$ 属于正太分布，进而代码实现如下：

```python
class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切
```

## DDPG算法

DDPG算法主要也包括两个功能，一个是选择动作，另外一个是更新策略，首先看选择动作：

```python
def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]
```

由于DDPG是直接从演员网络取得动作，所以这里不用$\epsilon-greedy$策略。在更新策略函数中，也会跟DQN稍有不同，并且加入软更新：

```python
def update(self):
        if len(self.memory) < self.batch_size: # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
       
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
```

## 结果分析

实现算法之后，我们先看看训练效果：

![train_rewards_curve_cn](../../easy_rl_book/res/ch12/assets/train_rewards_curve_cn-1760758.png)

可以看到算法整体上是达到收敛了的，但是稳定状态下波动还比较大，依然有提升的空间，限于笔者的精力，这里只是帮助赌注实现一个基础的代码演示，想要使得算法调到最优感兴趣的读者可以多思考实现。我们再来看看测试的结果：

![eval_rewards_curve_cn](../../easy_rl_book/res/ch12/assets/eval_rewards_curve_cn-1760950.png)

从图中看出测试的平均奖励在-150左右，但其实训练的时候平均的稳态奖励在-300左右，这是因为测试的时候我们舍去了OU噪声的缘故。