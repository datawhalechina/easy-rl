# Chapter7 Q-learning-Double DQN

## 1 Keywords

- **Double DQN：** 在Double DQN中存在有两个 Q-network，首先，第一个 Q-network，决定的是哪一个 action 的 Q value 最大，从而决定了你的action。另一方面， Q value 是用 $Q'$ 算出来的，这样就可以避免 over estimate 的问题。具体来说，假设我们有两个 Q-function，假设第一个Q-function 它高估了它现在选出来的action a，那没关系，只要第二个Q-function $Q'$ 没有高估这个action a 的值，那你算出来的，就还是正常的值。
- **Dueling DQN：** 将原来的DQN的计算过程分为**两个path**。对于第一个path，会计算一个于input state有关的一个标量 $V(s)$；对于第二个path，会计算出一个vector $A(s,a)$ ，其对应每一个action。最后的网络是将两个path的结果相加，得到我们最终需要的Q value。用一个公式表示也就是 $Q(s,a)=V(s)+A(s,a)$ 。 
- **Prioritized Experience Replay （优先经验回放）：** 这个方法是为了解决我们在chapter6中提出的**Experience Replay（经验回放）**方法不足进一步优化提出的。我们在使用Experience Replay时是uniformly取出的experience buffer中的sample data，这里并没有考虑数据间的权重大小。例如，我们应该将那些train的效果不好的data对应的权重加大，即其应该有更大的概率被sample到。综上， prioritized experience replay 不仅改变了 sample data 的 distribution，还改变了 training process。
- **Noisy Net：** 其在每一个episode 开始的时候，即要和环境互动的时候，将原来的Q-function 的每一个参数上面加上一个Gaussian noise。那你就把原来的Q-function 变成$\tilde{Q}$ ，即**Noisy Q-function**。同样的我们把每一个network的权重等参数都加上一个Gaussian noise，就得到一个新的network $\tilde{Q}$。我们会使用这个新的network从与环境互动开始到互动结束。
- **Distributional Q-function：** 对于DQN进行model distribution。将最终的网络的output的每一类别的action再进行distribution。
- **Rainbow：** 也就是将我们这两节内容所有的七个tips综合起来的方法，7个方法分别包括：DQN、DDQN、Prioritized DDQN、Dueling DDQN、A3C、Distributional DQN、Noisy DQN，进而考察每一个方法的贡献度或者是否对于与环境的交互式正反馈的。

## 2 Questions

- 为什么传统的DQN的效果并不好？参考公式 $Q(s_t ,a_t)=r_t+\max_{a}Q(s_{t+1},a)$ 

  答：因为实际上在做的时候，是要让左边这个式子跟右边这个 target 越接近越好。比较容易可以发现target 的值很容易一不小心就被设得太高。因为在算这个 target 的时候，我们实际上在做的事情是看哪一个a 可以得到最大的Q value，就把它加上去，就变成我们的target。

  举例来说，现在有 4 个 actions，本来其实它们得到的值都是差不多的，它们得到的reward 都是差不多的。但是在estimate 的时候，那毕竟是个network。所以estimate 的时候是有误差的。所以假设今天是第一个action它被高估了，假设绿色的东西代表是被高估的量，它被高估了，那这个target 就会选这个action。然后就会选这个高估的Q value来加上$r_t$，来当作你的target。如果第4 个action 被高估了，那就会选第4 个action 来加上$r_t$ 来当作你的target value。所以你总是会选那个Q value 被高估的，你总是会选那个reward 被高估的action 当作这个max 的结果去加上$r_t$ 当作你的target。所以你的target 总是太大。

- 接着上个思考题，我们应该怎么解决target 总是太大的问题呢？

  答： 我们可以使用Double DQN解决这个问题。首先，在 Double DQN 里面，选 action 的 Q-function 跟算 value 的 Q-function不同。在原来的DQN 里面，你穷举所有的 a，把每一个a 都带进去， 看哪一个 a 可以给你的 Q value 最高，那你就把那个 Q value 加上$r_t$。但是在 Double DQN 里面，你**有两个 Q-network**，第一个 Q-network，决定哪一个 action 的 Q value 最大，你用第一个 Q-network 去带入所有的 a，去看看哪一个Q value 最大。然后你决定你的action 以后，你的 Q value 是用 $Q'$ 算出来的，这样子有什么好处呢？为什么这样就可以避免 over estimate 的问题呢？因为今天假设我们有两个 Q-function，假设第一个Q-function 它高估了它现在选出来的action a，那没关系，只要第二个Q-function $Q'$ 没有高估这个action a 的值，那你算出来的，就还是正常的值。假设反过来是 $Q'$ 高估了某一个action 的值，那也没差， 因为反正只要前面这个Q 不要选那个action 出来就没事了。

- 哪来 Q  跟 $Q'$ 呢？哪来两个 network 呢？

  答：在实现上，你有两个 Q-network， 一个是 target 的 Q-network，一个是真正你会 update 的 Q-network。所以在 Double DQN 里面，你的实现方法会是拿你会 update 参数的那个 Q-network 去选action，然后你拿target 的network，那个固定住不动的network 去算value。而 Double DQN 相较于原来的 DQN 的更改是最少的，它几乎没有增加任何的运算量，连新的network 都不用，因为你原来就有两个network 了。你唯一要做的事情只有，本来你在找最大的a 的时候，你在决定这个a 要放哪一个的时候，你是用$Q'$ 来算，你是用target network 来算，现在改成用另外一个会 update 的 Q-network 来算。

- 如何理解Dueling DQN的模型变化带来的好处？

  答：对于我们的 $Q(s,a)$ 其对应的state由于为table的形式，所以是离散的，而实际中的state不是离散的。对于 $Q(s,a)$ 的计算公式， $Q(s,a)=V(s)+A(s,a)$ 。其中的 $V(s)$ 是对于不同的state都有值，对于 $A(s,a)$ 对于不同的state都有不同的action对应的值。所以本质上来说，我们最终的矩阵 $Q(s,a)$ 的结果是将每一个 $V(s)$ 加到矩阵 $A(s,a)$ 中得到的。从模型的角度考虑，我们的network直接改变的 $Q(s,a)$ 而是 更改的 $V、A$ 。但是有时我们update时不一定会将 $V(s)$ 和 $Q(s,a)$ 都更新。我们将其分成两个path后，我们就不需要将所有的state-action pair都sample一遍，我们可以使用更高效的estimate Q value方法将最终的 $Q(s,a)$ 计算出来。

- 使用MC和TD平衡方法的优劣分别有哪些？

  答：

  - 优势：因为我们现在 sample 了比较多的step，之前是只sample 了一个step， 所以某一个step 得到的data 是真实值，接下来都是Q value 估测出来的。现在sample 比较多step，sample N 个step 才估测value，所以估测的部分所造成的影响就会比小。
  - 劣势：因为我们的 reward 比较多，当我们把 N 步的 reward 加起来，对应的 variance 就会比较大。但是我们可以选择通过调整 N 值，去在variance 跟不精确的 Q 之间取得一个平衡。这里介绍的参数 N 就是一个hyper parameter，你要调这个N 到底是多少，你是要多 sample 三步，还是多 sample 五步。



## 3 Something About Interview

- 高冷的面试官：DQN都有哪些变种？引入状态奖励的是哪种？

  答：DQN三个经典的变种：Double DQN、Dueling DQN、Prioritized Replay Buffer。

  - Double-DQN：将动作选择和价值估计分开，避免价值过高估计。
  - Dueling-DQN：将Q值分解为状态价值和优势函数，得到更多有用信息。
  - Prioritized Replay Buffer：将经验池中的经验按照优先级进行采样。

- 简述double DQN原理？

  答：DQN由于总是选择当前值函数最大的动作值函数来更新当前的动作值函数，因此存在着过估计问题（估计的值函数大于真实的值函数）。为了解耦这两个过程，double DQN 使用了两个值网络，一个网络用来执行动作选择，然后用另一个值函数对一个的动作值更新当前网络。

- 高冷的面试官：请问Dueling DQN模型有什么优势呢？

  答：对于我们的 $Q(s,a)$ 其对应的state由于为table的形式，所以是离散的，而实际中的state不是离散的。对于 $Q(s,a)$ 的计算公式， $Q(s,a)=V(s)+A(s,a)$ 。其中的 $V(s)$ 是对于不同的state都有值，对于 $A(s,a)$ 对于不同的state都有不同的action对应的值。所以本质上来说，我们最终的矩阵 $Q(s,a)$ 的结果是将每一个 $V(s)$ 加到矩阵 $A(s,a)$ 中得到的。从模型的角度考虑，我们的network直接改变的 $Q(s,a)$ 而是更改的 $V、A$ 。但是有时我们update时不一定会将 $V(s)$ 和 $Q(s,a)$ 都更新。我们将其分成两个path后，我们就不需要将所有的state-action pair都sample一遍，我们可以使用更高效的estimate Q value方法将最终的 $Q(s,a)$ 计算出来。
