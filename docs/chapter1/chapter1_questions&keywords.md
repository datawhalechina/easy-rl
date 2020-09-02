## Chapter1  强化学习概述

#### 1 关键词

- **强化学习（Reinforcement Learning）**：Agent可以在与复杂且不确定的Environment进行交互时，尝试使所获得的Reward最大化的计算算法。
- **Action**: Environment接收到的Agent当前状态的输出。
- **State**：Agent从Environment中获取到的状态。
- **Reward**：Agent从Environment中获取的反馈信号，这个信号指定了Agent 在某一步采取了某个策略是否得到奖励。
- **Exploration**：在当前的情况下，继续尝试新的Action，其有可能会使你得到更高的这个奖励，也有可能使你一无所有。
- **Exploitation**：在当前的情况下，继续尝试已知可以获得最大Reward的过程，那你就重复执行这个 Action 就可以了。
- **深度强化学习（Deep Reinforcement Learning）**：不需要手工设计特征，仅需要输入State让系统直接输出Action的一个end-to-end training的强化学习方法。通常使用神经网络来拟合 value function 或 policy network。
- **Full observability、fully observed和partially observed**：当 Agent 的状态跟Environment的状态等价的时候，我们就说现在Environment是full observability（全部可观测），当 Agent 能够观察到Environment的所有状态时，我们称这个环境是fully observed（完全可观测）。一般我们的Agent不能观察到Environment的所有状态时，我们称这个环境是partially observed（部分可观测）。
- **POMDP（Partially Observable Markov Decision Processes）**：部分可观测马尔可夫决策过程，即马尔可夫决策过程的泛化。POMDP 依然具有马尔可夫性质，但是假设智能体无法感知环境的状态 $s$，只能知道部分观测值 $o$。
- **Action space（discrete action spaces and continuous action spaces）**：在给定的Environment中，有效动作的集合经常被称为动作空间（Action space），Agent 的动作数量是有限的动作空间为离散动作空间（discrete action spaces），反之，称为连续动作空间（continuous action spaces）。
- **policy-based（基于策略的）**：智能体会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励。
- **valued-based（基于价值的）**：智能体不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。
- **model-based（有模型结构）**：Agent通过学习状态的转移来采取措施。
- **model-free（无模型结构）**：Agent没有去直接估计状态的转移，也没有得到Environment的具体转移变量。它通过学习 value function 和 policy function 进行决策。

#### 2 思考题

- 强化学习的基本结构是什么？

  答：本质上是Agent和Environment间的交互。具体地，当Agent在Environment中得到当前时刻的State，Agent会基于此状态输出一个Action。然后这个Action会加入到Environment中去并输出下一个State和当前的这个Action得到的Reward。Agent 在Environment里面存在的目的就是为了极大它的期望积累的Reward。

- 强化学习相对于监督学习为什么训练会更加困难？（强化学习的特征）

  答：

  1. 强化学习处理的多是序列数据，其很难像监督学习的样本一样满足**IID（独立同分布）**条件。

  2. 强化学习有奖励的延迟（Delay Reward），即在Agent的action作用在Environment中时，Environment对于Agent的State的**奖励的延迟**（Delayed Reward），使得反馈不及时。
  3. 相比于监督学习有正确的label，可以通过其修正自己的预测，强化学习相当于一个“试错”的过程，其完全根据Environment的“**反馈**”更新对自己最有利的Action。

- 强化学习的基本特征有哪些？

  答： 

  1. 有**trial-and-error exploration**的过程，即需要通过探索Environment来获取对这个Environment的理解。

  2. 强化学习的Agent 会从Environment里面获得**延迟**的Reward。
  3. 强化学习的训练过程中**时间**非常重要，因为数据都是有时间关联的，而不是像监督学习一样是IID分布的。
  4.  强化学习中Agent的Action会**影响**它随后得到的**反馈**。

- 近几年强化学习发展迅速的原因？

  答：

  1. **算力（GPU、TPU）的提升**，我们可以更快地做更多的 trial-and-error 的尝试来使得 Agent 在Environment里面获得很多信息，取得很大的Reward。

  2.  我们有了深度强化学习这样一个端到端的训练方法，可以把特征提取和价值估计或者决策一起优化，这样就可以得到一个更强的决策网络。

- 状态和观测有什么关系？

  答：状态（state）是对世界的**完整描述**，不会隐藏世界的信息。观测（observation）是对状态的**部分描述**，可能会遗漏一些信息。在深度强化学习中，我们几乎总是用一个实值向量、矩阵或者更高阶的张量来表示状态和观测。

- 对于一个强化学习 Agent，它由什么组成？

  答：

  1. **策略函数（policy function）**，Agent 会用这个函数来选取它下一步的动作，包括**随机性策略（stochastic policy）**和**确定性策略（deterministic policy）**。

  2. **价值函数（value function）**，我们用价值函数来对当前状态进行估价，它就是说你进入现在这个状态，到底可以对你后面的收益带来多大的影响。当这个价值函数大的时候，说明你进入这个状态越有利。

  3. **模型（model）**，其表示了 Agent 对这个Environment的状态进行的理解，它决定了这个系统是如何进行的。

- 根据强化学习 Agent 的不同，我们可以将其分为哪几类？

  答：

  1. **基于价值函数的 Agent**。 显式学习的就是价值函数，隐式地学习了它的策略。因为这个策略是从我们学到的价值函数里面推算出来的。
  2. **基于策略的 Agent**。它直接去学习 policy，就是说你直接给它一个 state，它就会输出这个动作的概率。然后在这个 policy-based agent 里面并没有去学习它的价值函数。
  3. 然后另外还有一种 Agent 是把这两者结合。把 value-based 和 policy-based 结合起来就有了 **Actor-Critic agent**。这一类 Agent 就把它的策略函数和价值函数都学习了，然后通过两者的交互得到一个最佳的行为。

- 基于策略迭代和基于价值迭代的强化学习方法有什么区别?

  答：基于策略迭代的强化学习方法，智能体会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励；基于价值迭代的强化学习方法，智能体不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。基于价值迭代的方法只能应用在不连续的、离散的环境下（如围棋或某些游戏领域），对于行为集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时基于策略迭代的方法能够根据设定的策略来选择连续的动作)；基于价值迭代的强化学习算法有 Q-learning、 Sarsa 等，而基于策略迭代的强化学习算法有策略梯度算法等。此外， Actor-Critic 算法同时使用策略和价值评估来做出决策，其中，智能体会根据策略做出动作，而价值函数会对做出的动作给出价值，这样可以在原有的策略梯度算法的基础上加速学习过程，取得更好的效果。

- 有模型（model-based）学习和免模型（model-free）学习有什么区别？

  答：针对是否需要对真实环境建模，强化学习可以分为有模型学习和免模型学习。有模型学习是指根据环境中的经验，构建一个虚拟世界，同时在真实环境和虚拟世界中学习；免模型学习是指不对环境进行建模，直接与真实环境进行交互来学习到最优策略。总的来说，有模型学习相比于免模型学习仅仅多出一个步骤，即对真实环境进行建模。免模型学习通常属于数据驱动型方法，需要大量的采样来估计状态、动作及奖励函数，从而优化动作策略。免模型学习的泛化性要优于有模型学习，原因是有模型学习算需要对真实环境进行建模，并且虚拟世界与真实环境之间可能还有差异，这限制了有模型学习算法的泛化性。

- 强化学习的通俗理解

  答：environment 跟 reward function 不是我们可以控制的，environment 跟 reward function 是在开始学习之前，就已经事先给定的。我们唯一能做的事情是调整 actor 里面的 policy，使得 actor 可以得到最大的 reward。Actor 里面会有一个 policy， 这个policy 决定了actor 的行为。Policy 就是给一个外界的输入，然后它会输出 actor 现在应该要执行的行为。
  

