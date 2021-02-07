# Chapter9 Actor-Critic

## 1 Keywords

- **A2C：** Advantage Actor-Critic的缩写，一种Actor-Critic方法。

- **A3C：** Asynchronous（异步的）Advantage Actor-Critic的缩写，一种改进的Actor-Critic方法，通过异步的操作，进行RL模型训练的加速。
-  **Pathwise Derivative Policy Gradient：** 其为使用 Q-learning 解 continuous action 的方法，也是一种 Actor-Critic 方法。其会对于actor提供value最大的action，而不仅仅是提供某一个action的好坏程度。

## 2 Questions

- 整个Advantage actor-critic（A2C）算法的工作流程是怎样的？

  答：在传统的方法中，我们有一个policy $\pi$ 以及一个初始的actor与environment去做互动，收集数据以及反馈。通过这些每一步得到的数据与反馈，我们就要进一步更新我们的policy $\pi$ ，通常我们所使用的方式是policy gradient。但是对于actor-critic方法，我们不是直接使用每一步得到的数据和反馈进行policy $\pi$ 的更新，而是使用这些数据进行 estimate value function，这里我们通常使用的算法包括前几个chapters重点介绍的TD和MC等算法以及他们的优化算法。接下来我们再基于value function来更新我们的policy，公式如下：
  $$
  \nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}}\left(r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)-V^{\pi}\left(s_{t}^{n}\right)\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
  $$
  其中，上式中的 $r_{t}^{n}+V^{\pi}\left(s_{t+1}^{n}\right)-V^{\pi}\left(s_{t}^{n}\right)$ 我们称为Advantage function，我们通过上式得到新的policy后，再去与environment进行交互，然后再重复我们的estimate value function的操作，再用value function来更新我们的policy。以上的整个方法我们称为Advantage Actor-Critic。

- 在实现 Actor-Critic 的时候，有哪些我们用到的tips?

  答：与我们上一章讲述的东西有关：

  1. **estimate 两个 network：** 一个是estimate V function，另外一个是 policy 的 network，也就是你的 actor。 V-network的input 是一个 state，output 是一个 scalar。然后 actor 这个 network的input 是一个 state，output 是一个 action 的 distribution。这两个 network，actor 和 critic 的 input 都是 s，所以它们前面几个 layer，其实是可以 share 的。尤其是假设你今天是玩 Atari 游戏，input 都是 image。那 input 那个 image 都非常复杂，image 很大，通常前面都会用一些 CNN 来处理，把那些 image 抽象成 high level 的 information，所以对 actor 跟 critic 来说是可以共用的。我们可以让 actor 跟 critic 的前面几个 layer 共用同一组参数。那这一组参数可能是 CNN。先把 input 的 pixel 变成比较 high level 的信息，然后再给 actor 去决定说它要采取什么样的行为，给这个 critic，给 value function 去计算 expected reward。
  2. **exploration 机制：** 其目的是对policy $\pi$ 的 output 的分布进行一个限制，从而使得 distribution 的 entropy 不要太小，即希望不同的 action 被采用的机率平均一点。这样在 testing 的时候，它才会多尝试各种不同的 action，才会把这个环境探索的比较好，才会得到比较好的结果。

- A3C（Asynchronous Advantage Actor-Critic）在训练是回有很多的worker进行异步的工作，最后再讲他们所获得的“结果”再集合到一起。那么其具体的如何运作的呢？

  答：A3C一开始会有一个 global network。它们有包含 policy 的部分和 value 的部分，假设它的参数就是 $\theta_1$。对于每一个 worker 都用一张 CPU 训练（举例子说明），第一个 worker 就把 global network 的参数 copy 过来，每一个 worker 工作前都会global network 的参数 copy 过来。然后这个worker就要去跟environment进行交互，每一个 actor 去跟environment做互动后，就会计算出 gradient并且更新global network的参数。这里要注意的是，所有的 actor 都是平行跑的、之间没有交叉。所以每个worker都是在global network“要”了一个参数以后，做完就把参数传回去。所以当第一个 worker 做完想要把参数传回去的时候，本来它要的参数是 $\theta_1$，等它要把 gradient 传回去的时候。可能别人已经把原来的参数覆盖掉，变成 $\theta_2$了。但是没有关系，它一样会把这个 gradient 就覆盖过去就是了。

- 对比经典的Q-learning算法，我们的Pathwise Derivative Policy Gradient有哪些改进之处？

  答：

  1. 首先，把 $Q(s,a)$ 换成 了 $\pi$，之前是用 $Q(s,a)$ 来决定在 state $s_t$ 产生那一个 action, $a_{t}$ 现在是直接用 $\pi$ 。原先我们需要解 argmax 的问题，现在我们直接训练了一个 actor。这个 actor input $s_t$ 就会告诉我们应该采取哪一个 $a_{t}$。综上，本来 input $s_t$，采取哪一个 $a_t$，是 $Q(s,a)$ 决定的。在 Pathwise Derivative Policy Gradient 里面，我们会直接用 $\pi$ 来决定。
  2. 另外，原本是要计算在 $s_{i+1}$ 时对应的 policy 采取的 action a 会得到多少的 Q value，那你会采取让 $\hat{Q}$ 最大的那个 action a。现在因为我们不需要再解argmax 的问题。所以现在我们就直接把 $s_{i+1}$ 代入到 policy $\pi$ 里面，直接就会得到在 $s_{i+1}$ 下，哪一个 action 会给我们最大的 Q value，那你在这边就会 take 那一个 action。在 Q-function 里面，有两个 Q network，一个是真正的 Q network，另外一个是 target Q network。那实际上你在 implement 这个 algorithm 的时候，你也会有两个 actor，你会有一个真正要 learn 的 actor $\pi$，你会有一个 target actor $\hat{\pi}$ 。但现在因为哪一个 action a 可以让 $\hat{Q}$ 最大这件事情已经被用那个 policy 取代掉了，所以我们要知道哪一个 action a 可以让 $\hat{Q}$ 最大，就直接把那个 state 带到 $\hat{\pi}$ 里面，看它得到哪一个 a，就用那一个 a，其也就是会让 $\hat{Q}(s,a)$ 的值最大的那个 a 。
  3. 还有，之前只要 learn Q，现在你多 learn 一个 $\pi$，其目的在于maximize Q-function，希望你得到的这个 actor，它可以让你的 Q-function output 越大越好，这个跟 learn GAN 里面的 generator 的概念类似。
  4. 最后，与原来的 Q-function 一样。我们要把 target 的 Q-network 取代掉，你现在也要把 target policy 取代掉。
