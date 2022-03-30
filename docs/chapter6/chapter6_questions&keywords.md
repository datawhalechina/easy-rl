# Chapter6 Q-learning-State Value Function

## 1 Keywords

- **DQN(Deep Q-Network)：**  基于深度学习的Q-learninyang算法，其结合了 Value Function Approximation（价值函数近似）与神经网络技术，并采用了目标网络（Target Network）和经验回放（Experience Replay）等方法进行网络的训练。
- **State-value Function：** 本质是一种critic。其输入为actor某一时刻的state，对应的输出为一个标量，即当actor在对应的state时，预期的到过程结束时间段中获得的value的数值。
- **State-value Function Bellman Equation：** 基于state-value function的Bellman Equation，它表示在状态 $s_t$ 下带来的累积奖励 $G_t$ 的期望。
- **Q-function:** 其也被称为state-action value function。其input 是一个 state 跟 action 的 pair，即在某一个 state 采取某一个action，假设我们都使用 actor $\pi$ ，得到的 accumulated reward 的期望值有多大。
- **Target Network：** 为了解决在基于TD的Network的问题时，优化目标 $\mathrm{Q}^{\pi}\left(s_{t}, a_{t}\right) 
  =r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 左右两侧会同时变化使得训练过程不稳定，从而增大regression的难度。target network选择将上式的右部分即 $r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 固定，通过改变上式左部分的network的参数，进行regression，这也是一个DQN中比较重要的tip。
- **Exploration：**  在我们使用Q-function的时候，我们的policy完全取决于Q-function，有可能导致出现对应的action是固定的某几个数值的情况，而不像policy gradient中的output为随机的，我们再从随机的distribution中sample选择action。这样会导致我们继续训练的input的值一样，从而“加重”output的固定性，导致整个模型的表达能力的急剧下降，这也就是`探索-利用窘境难题(Exploration-Exploitation dilemma)`。所以我们使用`Epsilon Greedy`和 `Boltzmann Exploration`等Exploration方法进行优化。
- **Experience Replay（经验回放）：**  其会构建一个Replay Buffer（Replay Memory），用来保存许多data，每一个data的形式如下：在某一个 state $s_t$，采取某一个action $a_t$，得到了 reward $r_t$，然后跳到 state $s_{t+1}$。我们使用 $\pi$ 去跟环境互动很多次，把收集到的数据都放到这个 replay buffer 中。当我们的buffer”装满“后，就会自动删去最早进入buffer的data。在训练时，对于每一轮迭代都有相对应的batch（与我们训练普通的Network一样通过sample得到），然后用这个batch中的data去update我们的Q-function。综上，Q-function再sample和训练的时候，会用到过去的经验数据，所以这里称这个方法为Experience Replay，其也是DQN中比较重要的tip。

## 2 Questions

- 为什么在DQN中采用价值函数近似（Value Function Approximation）的表示方法？

  答：首先DQN为基于深度学习的Q-learning算法，而在Q-learning中，我们使用表格来存储每一个state下action的reward，即我们前面所讲的状态-动作值函数 $Q(s,a)$ 。但是在我们的实际任务中，状态量通常数量巨大并且在连续的任务中，会遇到维度灾难的问题，所以使用真正的Value Function通常是不切实际的，所以使用了价值函数近似（Value Function Approximation）的表示方法。

- critic output通常与哪几个值直接相关？

  答：critic output与state和actor有关。我们在讨论output时通常是对于一个actor下来衡量一个state的好坏，也就是state value本质上来说是依赖于actor。不同的actor在相同的state下也会有不同的output。

- 我们通常怎么衡量state value function  $V^{\pi}(s)$ ?分别的优势和劣势有哪些？

  答：

  - **基于Monte-Carlo（MC）的方法** ：本质上就是让actor与environment做互动。critic根据”统计“的结果，将actor和state对应起来，即当actor如果看到某一state $s_a$ ，将预测接下来的accumulated reward有多大如果它看到 state $s_b$，接下来accumulated reward 会有多大。 但是因为其普适性不好，其需要把所有的state都匹配到，如果我们我们是做一个简单的贪吃蛇游戏等state有限的问题，还可以进行。但是如果我们做的是一个图片型的任务，我们几乎不可能将所有的state（对应每一帧的图像）的都”记录“下来。总之，其不能对于未出现过的input state进行对应的value的输出。
  - **基于MC的Network方法：** 为了解决上面描述的Monte-Carlo（MC）方法的不足，我们将其中的state value function  $V^{\pi}(s)$ 定义为一个Network，其可以对于从未出现过的input state，根据network的泛化和拟合能力，也可以”估测“出一个value output。
  - **基于Temporal-difference（时序差分）的Network方法，即TD based Network：** 与我们再前4章介绍的MC与TD的区别一样，这里两者的区别也相同。在 MC based 的方法中，每次我们都要算 accumulated reward，也就是从某一个 state $s_a$ 一直玩到游戏结束的时候，得到的所有 reward 的总和。所以要应用 MC based 方法时，我们必须至少把这个游戏玩到结束。但有些游戏非常的长，你要玩到游戏结束才能够 update network，花的时间太长了。因此我们会采用 TD based 的方法。TD based 的方法不需要把游戏玩到底，只要在游戏的某一个情况，某一个 state $s_t$ 的时候，采取 action $a_t$ 得到 reward $r_t$ ，跳到 state $s_{t+1}$，就可以应用 TD 的方法。公式与之前介绍的TD方法类似，即：$V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$。
  - **基于MC和基于TD的区别在于：** MC本身具有很大的随机性，我们可以将其 $G_a$  堪称一个random的变量，所以其最终的variance很大。而对于TD，其具有随机性的变量为 $r$ ,因为计算 $s_t$ 我们采取同一个 action，你得到的 reward 也不一定是一样的，所以对于TD来说，$r$ 是一个 random 变量。但是相对于MC的 $G_a$  的随机程度来说， $r$ 的随机性非常小，这是因为本身 $G_a$ 就是由很多的 $r$ 组合而成的。但另一个角度来说， 在TD中，我们的前提是 $r_t=V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right)$ ,但是我们通常无法保证 $V^{\pi}\left(s_{t+1}\right)、V^{\pi}\left(s_{t}\right)$ 计算的误差为零。所以当 $V^{\pi}\left(s_{t+1}\right)、V^{\pi}\left(s_{t}\right)$  计算的不准确的话，那应用上式得到的结果，其实也会是不准的。所以 MC 跟 TD各有优劣。
  - **目前， TD 的方法是比较常见的，MC 的方法其实是比较少用的。**

- 基于我们上面说的network（基于MC）的方法，我们怎么训练这个网络呢？或者我们应该将其看做ML中什么类型的问题呢？

  答：理想状态，我们期望对于一个input state输出其无误差的reward value。也就是说这个 value function 来说，如果 input 是 state $s_a$，正确的 output 应该是$G_a$。如果 input state $s_b$，正确的output 应该是value $G_b$。所以在训练的时候，其就是一个典型的ML中的回归问题（regression problem）。所以我们实际中需要输出的仅仅是一个非精确值，即你希望在 input $s_a$ 的时候，output value 跟 $G_a$ 越近越好，input $s_b$ 的时候，output value 跟 $G_b$ 越近越好。其训练方法，和我们在训练CNN、DNN时的方法类似，就不再一一赘述。

- 基于上面介绍的基于TD的network方法，具体地，我们应该怎么训练模型呢？

  答：核心的函数为 $V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$。我们将state $s_t$  作为input输入network 里，因为 $s_t$ 丢到 network 里面会得到output $V^{\pi}(s_t)$，同样将 $s_{t+1}$ 作为input输入 network 里面会得到$V^{\pi}(s_{t+1})$。同时核心函数：$V^{\pi}\left(s_{t}\right)=V^{\pi}\left(s_{t+1}\right)+r_{t}$  告诉我们，  $V^{\pi}(s_t)$ 减 $V^{\pi}(s_{t+1})$ 的值应该是 $r_t$。然后希望它们两个相减的 loss 跟 $r_t$ 尽可能地接近。这也就是我们这个network的优化目标或者说loss function。

- state-action value function（Q-function）和 state value function的有什么区别和联系？

  答：

  - state value function 的 input 是一个 state，它是根据 state 去计算出，看到这个state 以后的 expected accumulated reward 是多少。
  - state-action value function 的 input 是一个 state 跟 action 的 pair，即在某一个 state 采取某一个action，假设我们都使用 actor $\pi$ ，得到的 accumulated reward 的期望值有多大。

- Q-function的两种表示方法？

  答：

  - 当input 是 state和action的pair时，output 就是一个 scalar。
  - 当input 仅是一个 state时，output 就是好几个 value。

- 当我们有了Q-function后，我们怎么找到更好的策略 $\pi'$ 呢？或者说这个 $\pi'$ 本质来说是什么？

  答：首先， $\pi'$ 是由 $\pi^{\prime}(s)=\arg \max _{a} Q^{\pi}(s, a)$ 计算而得，其表示假设你已经 learn 出 $\pi$ 的Q-function，今天在某一个 state s，把所有可能的 action a 都一一带入这个 Q-function，看看说那一个 a 可以让 Q-function 的 value 最大，那这一个 action，就是 $\pi'$ 会采取的 action。所以根据上式决定的actoin的步骤一定比原来的 $\pi$ 要好，即$V^{\pi^{\prime}}(s) \geq V^{\pi}(s)$。

- 解决`探索-利用窘境(Exploration-Exploitation dilemma)`问题的Exploration的方法有哪些？他们具体的方法是怎样的？

  答：

  1. **Epsilon Greedy：** 我们有$1-\varepsilon$ 的机率，通常 $\varepsilon$ 很小，完全按照Q-function 来决定action。但是有 $\varepsilon$ 的机率是随机的。通常在实现上 $\varepsilon$ 会随着时间递减。也就是在最开始的时候。因为还不知道那个action 是比较好的，所以你会花比较大的力气在做 exploration。接下来随着training 的次数越来越多。已经比较确定说哪一个Q 是比较好的。你就会减少你的exploration，你会把 $\varepsilon$ 的值变小，主要根据Q-function 来决定你的action，比较少做random，这是**Epsilon Greedy**。
  2. **Boltzmann Exploration：** 这个方法就比较像是 policy gradient。在 policy gradient 里面network 的output 是一个 expected action space 上面的一个的 probability distribution。再根据 probability distribution 去做 sample。所以也可以根据Q value 去定一个 probability distribution，假设某一个 action 的 Q value 越大，代表它越好，我们采取这个 action 的机率就越高。这是**Boltzmann Exploration**。

- 我们使用Experience Replay（经验回放）有什么好处？

  答：

  1. 首先，在强化学习的整个过程中， 最花时间的 step 是在跟环境做互动，使用GPU乃至TPU加速来训练 network 相对来说是比较快的。而用 replay buffer 可以减少跟环境做互动的次数，因为在训练的时候，我们的 experience 不需要通通来自于某一个policy（或者当前时刻的policy）。一些过去的 policy 所得到的 experience 可以放在 buffer 里面被使用很多次，被反复的再利用，这样让你的 sample 到 experience 的利用是高效的。
  2. 另外，在训练网络的时候，其实我们希望一个 batch 里面的 data 越 diverse 越好。如果你的 batch 里面的 data 都是同样性质的，我们的训练出的模型拟合能力不会很乐观。如果 batch 里面都是一样的 data，你 train 的时候，performance 会比较差。我们希望 batch data 越 diverse 越好。那如果 buffer 里面的那些 experience 通通来自于不同的 policy ，那你 sample 到的一个 batch 里面的 data 会是比较 diverse 。这样可以保证我们模型的性能至少不会很差。

- 在Experience Replay中我们是要观察 $\pi$ 的 value，里面混杂了一些不是 $\pi$ 的 experience ，这会有影响吗？

  答：没关系。这并不是因为过去的 $\pi$ 跟现在的 $\pi$ 很像， 就算过去的$\pi$ 没有很像，其实也是没有关系的。主要的原因是我们并不是去sample 一个trajectory，我们只sample 了一个experience，所以跟是不是 off-policy 这件事是没有关系的。就算是off-policy，就算是这些 experience 不是来自于 $\pi$，我们其实还是可以拿这些 experience 来估测 $Q^{\pi}(s,a)$。

- DQN（Deep Q-learning）和Q-learning有什么异同点？

  答：整体来说，从名称就可以看出，两者的目标价值以及价值的update方式基本相同，另外一方面，不同点在于：

  - 首先，DQN 将 Q-learning 与深度学习结合，用深度网络来近似动作价值函数，而 Q-learning 则是采用表格存储。
  - DQN 采用了我们前面所描述的经验回放（Experience Replay）训练方法，从历史数据中随机采样，而 Q-learning 直接采用下一个状态的数据进行学习。


## 3 Something About Interview

- 高冷的面试官：请问DQN（Deep Q-Network）是什么？其两个关键性的技巧分别是什么？

  答：Deep Q-Network是基于深度学习的Q-learning算法，其结合了 Value Function Approximation（价值函数近似）与神经网络技术，并采用了目标网络（Target Network）和经验回放（Experience Replay）的方法进行网络的训练。

- 高冷的面试官：接上题，DQN中的两个trick：目标网络和experience replay的具体作用是什么呢？

  答：在DQN中某个动作值函数的更新依赖于其他动作值函数。如果我们一直更新值网络的参数，会导致
  更新目标不断变化，也就是我们在追逐一个不断变化的目标，这样势必会不太稳定。 为了解决在基于TD的Network的问题时，优化目标 $\mathrm{Q}^{\pi}\left(s_{t}, a_{t}\right) =r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 左右两侧会同时变化使得训练过程不稳定，从而增大regression的难度。target network选择将上式的右部分即 $r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 固定，通过改变上式左部分的network的参数，进行regression。对于经验回放，其会构建一个Replay Buffer（Replay Memory），用来保存许多data，每一个data的形式如下：在某一个 state $s_t$，采取某一个action $a_t$，得到了 reward $r_t$，然后跳到 state $s_{t+1}$。我们使用 $\pi$ 去跟环境互动很多次，把收集到的数据都放到这个 replay buffer 中。当我们的buffer”装满“后，就会自动删去最早进入buffer的data。在训练时，对于每一轮迭代都有相对应的batch（与我们训练普通的Network一样通过sample得到），然后用这个batch中的data去update我们的Q-function。也就是，Q-function再sample和训练的时候，会用到过去的经验数据，也可以消除样本之间的相关性。

- 高冷的面试官：DQN（Deep Q-learning）和Q-learning有什么异同点？

  答：整体来说，从名称就可以看出，两者的目标价值以及价值的update方式基本相同，另外一方面，不同点在于：

  - 首先，DQN 将 Q-learning 与深度学习结合，用深度网络来近似动作价值函数，而 Q-learning 则是采用表格存储。
  - DQN 采用了我们前面所描述的经验回放（Experience Replay）训练方法，从历史数据中随机采样，而 Q-learning 直接采用下一个状态的数据进行学习。

- 高冷的面试官：请问，随机性策略和确定性策略有什么区别吗？

  答：随机策略表示为某个状态下动作取值的分布，确定性策略在每个状态只有一个确定的动作可以选。
  从熵的角度来说，确定性策略的熵为0，没有任何随机性。随机策略有利于我们进行适度的探索，确定
  性策略的探索问题更为严峻。

- 高冷的面试官：请问不打破数据相关性，神经网络的训练效果为什么就不好？

  答：在神经网络中通常使用随机梯度下降法。随机的意思是我们随机选择一些样本来增量式的估计梯度，比如常用的
  采用batch训练。如果样本是相关的，那就意味着前后两个batch的很可能也是相关的，那么估计的梯度也会呈现
  出某种相关性。如果不幸的情况下，后面的梯度估计可能会抵消掉前面的梯度量。从而使得训练难以收敛。
