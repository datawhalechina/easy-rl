# Chapter3 表格型方法 

## 1 Keywords

- **P函数和R函数：** P函数反应的是状态转移的概率，即反应的环境的随机性，R函数就是Reward function。但是我们通常处于一个未知的环境（即P函数和R函数是未知的）。
- **Q表格型表示方法：** 表示形式是一种表格形式，其中横坐标为 action（agent）的行为，纵坐标是环境的state，其对应着每一个时刻agent和环境的情况，并通过对应的reward反馈去做选择。一般情况下，Q表格是一个已经训练好的表格，不过，我们也可以每进行一步，就更新一下Q表格，然后用下一个状态的Q值来更新这个状态的Q值（即时序差分方法）。
- **时序差分（Temporal Difference）：** 一种Q函数（Q值）的更新方式，也就是可以拿下一步的 Q 值 $Q(S_{t+_1},A_{t+1})$ 来更新我这一步的 Q 值 $Q(S_t,A_t)$ 。完整的计算公式如下：$Q(S_t,A_t) \larr Q(S_t,A_t) + \alpha [R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$

- **SARSA算法：** 一种更新前一时刻状态的单步更新的强化学习算法，也是一种on-policy策略。该算法由于每次更新值函数需要知道前一步的状态(state)，前一步的动作(action)、奖励(reward)、当前状态(state)、将要执行的动作(action)，即 $(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$ 这几个值，所以被称为SARSA算法。agent每进行一次循环，都会用 $(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$ 对于前一步的Q值（函数）进行一次更新。

## 2 Questions

- 构成强化学习MDP的四元组有哪些变量？

  答：状态、动作、状态转移概率和奖励，分别对应（S，A，P，R），后面有可能会加上个衰减因子构成五元组。

- 基于以上的描述所构成的强化学习的“学习”流程。

  答：强化学习要像人类一样去学习了，人类学习的话就是一条路一条路的去尝试一下，先走一条路，我看看结果到底是什么。多试几次，只要能一直走下去的，我们其实可以慢慢的了解哪个状态会更好。我们用价值函数 $V(s)$ 来代表这个状态是好的还是坏的。然后用这个 Q 函数来判断说在什么状态下做什么动作能够拿到最大奖励，我们用 Q 函数来表示这个状态-动作值。

- 基于SARSA算法的agent的学习过程。

  答：我们现在有环境，有agent。每交互一次以后，我们的agent会向环境输出action，接着环境会反馈给agent当前时刻的state和reward。那么agent此时会实现两个方法：
  
  1.使用已经训练好的Q表格，对应环境反馈的state和reward选取对应的action进行输出。
  
  2.我们已经拥有了$(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$  这几个值，并直接使用 $A_{t+1}$ 去更新我们的Q表格。

- Q-learning和Sarsa的区别？

  答：Sarsa算法是Q-learning算法的改进。（这句话出自「神经网络与深度学习」的第 342 页）（可参考SARSA「on-line q-learning using connectionist systems」的 abstract 部分）

  1. 首先，Q-learning 是 off-policy 的时序差分学习方法，Sarsa 是 on-policy 的时序差分学习方法。

  2. 其次，Sarsa 在更新 Q 表格的时候，它用到的 A' 。我要获取下一个 Q 值的时候，A' 是下一个 step 一定会执行的 action 。这个 action 有可能是 $\varepsilon$-greddy 方法 sample 出来的值，也有可能是 max Q 对应的 action，也有可能是随机动作。但是就是它实实在在执行了的那个动作。

  3. 但是 Q-learning 在更新 Q 表格的时候，它用到这个的 Q 值 $Q(S',a')$ 对应的那个 action ，它不一定是下一个 step 会执行的实际的 action，因为你下一个实际会执行的那个 action 可能会探索。Q-learning 默认的 action 不是通过 behavior policy 来选取的，它是默认 A' 为最优策略选的动作，所以 Q-learning 在学习的时候，不需要传入 A'，即 $a_{t+1}$  的值。

  4. 更新公式的对比（区别只在target计算这一部分）：

     - Sarsa的公式： $R_{t+1}+\gamma Q(S_{t+1}, A_{t+1})$ ；
     - Q-learning的公式：$R_{t+1}+\gamma  \underset{a}{\max} Q\left(S_{t+1}, a\right)$

     Sarsa 实际上都是用自己的策略产生了 S,A,R,S',A' 这一条轨迹。然后拿着 $Q(S_{t+1},A_{t+1})$ 去更新原本的 Q 值 $Q(S_t,A_t)$。 但是 Q-learning 并不需要知道，我实际上选择哪一个 action ，它默认下一个动作就是 Q 最大的那个动作。所以基于此，Sarsa的action通常会更加“保守”、“胆小”，而对应的Q-Learning的action会更加“莽撞”、“激进”。

- On-policy和 off-policy 的区别？

  答：

  1. Sarsa 就是一个典型的 on-policy 策略，它只用一个 $\pi$ ，为了兼顾探索和利用，所以它训练的时候会显得有点胆小怕事。它在解决悬崖问题的时候，会尽可能地离悬崖边上远远的，确保说哪怕自己不小心探索了一点了，也还是在安全区域内不不至于跳进悬崖。
  2. Q-learning 是一个比较典型的 off-policy 的策略，它有目标策略 target policy，一般用 $\pi$ 来表示。然后还有行为策略 behavior policy，用 $\mu$ 来表示。它分离了目标策略跟行为策略。Q-learning 就可以大胆地用 behavior policy 去探索得到的经验轨迹来去优化我的目标策略。这样子我更有可能去探索到最优的策略。
  3. 比较 Q-learning 和 Sarsa 的更新公式可以发现，Sarsa 并没有选取最大值的 max 操作。因此，Q-learning 是一个非常激进的算法，希望每一步都获得最大的利益；而 Sarsa 则相对非常保守，会选择一条相对安全的迭代路线。
