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


## 3 Something About Interview

- 高冷的面试官：同学，你能否简述on-policy和off-policy的区别？

  答： off-policy和on-policy的根本区别在于生成样本的policy和参数更新时的policy是否相同。对于on-policy，行为策略和要优化的策略是一个策略，更新了策略后，就用该策略的最新版本对于数据进行采样；对于off-policy，使用任意的一个行为策略来对于数据进行采样，并利用其更新目标策略。如果举例来说，Q-learning在计算下一状态的预期收益时使用了max操作，直接选择最优动作，而当前policy并不一定能选择到最优的action，因此这里生成样本的policy和学习时的policy不同，所以Q-learning为off-policy算法；相对应的SARAS则是基于当前的policy直接执行一次动作选择，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同，所以SARAS算法为on-policy算法。

- 高冷的面试官：小同学，能否讲一下Q-Learning，最好可以写出其 $Q(s,a)$ 的更新公式。另外，它是on-policy还是off-policy，为什么？

  答： Q-learning是通过计算最优动作值函数来求策略的一种时序差分的学习方法，其更新公式为：
  $$
  Q(s, a) \larr Q(s, a) + \alpha [r(s,a) + \gamma \max_{a'} Q(s', a') - Q(s, a)]
  $$
  其是off-policy的，由于是Q更新使用了下一个时刻的最大值，所以我们只关心哪个动作使得 $Q(s_{t+1}, a)$ 取得最大值，而实际到底采取了哪个动作（行为策略），并不关心。这表明优化策略并没有用到行为策略的数据，所以说它是 off-policy 的。

- 高冷的面试官：小朋友，能否讲一下SARSA，最好可以写出其Q(s,a)的更新公式。另外，它是on-policy还是off-policy，为什么？

  答：SARSA可以算是Q-learning的改进（这句话出自「神经网络与深度学习」的第 342 页）（可参考SARSA「on-line q-learning using connectionist systems」的 abstract 部分），其更新公式为：
  $$
  Q(s, a) \larr Q(s, a) + \alpha [r(s,a) + \gamma  Q(s', a') - Q(s, a)]
  $$
  其为on-policy的，SARSA必须执行两次动作得到 $(s,a,r,s',a') $才可以更新一次；而且 $a'$ 是在特定策略 $\pi$ 的指导下执行的动作，因此估计出来的 $Q(s,a)$ 是在该策略 $\pi$ 之下的Q-value，样本生成用的 $\pi$ 和估计的 $\pi$ 是同一个，因此是on-policy。

- 高冷的面试官：请问value-based和policy-based的区别是什么？

  答：

  1. 生成policy上的差异：前者确定，后者随机。Value-Base中的 action-value估计值最终会收敛到对应的true values（通常是不同的有限数，可以转化为0到1之间的概率），因此通常会获得一个确定的策略（deterministic policy）；而Policy-Based不会收敛到一个确定性的值，另外他们会趋向于生成optimal stochastic policy。如果optimal policy是deterministic的，那么optimal action对应的性能函数将远大于suboptimal actions对应的性能函数，性能函数的大小代表了概率的大小。
  2. 动作空间是否连续，前者离散，后者连续。Value-Base，对于连续动作空间问题，虽然可以将动作空间离散化处理，但离散间距的选取不易确定。过大的离散间距会导致算法取不到最优action，会在这附近徘徊，过小的离散间距会使得action的维度增大，会和高维度动作空间一样导致维度灾难，影响算法的速度；而Policy-Based适用于连续的动作空间，在连续的动作空间中，可以不用计算每个动作的概率，而是通过Gaussian distribution （正态分布）选择action。
  3. value-based，例如Q-learning，是通过求解最优值函数间接的求解最优策略；policy-based，例如REINFORCE，Monte-Carlo Policy Gradient，等方法直接将策略参数化，通过策略搜索，策略梯度或者进化方法来更新策略的参数以最大化回报。基于值函数的方法不易扩展到连续动作空间，并且当同时采用非线性近似、自举和离策略时会有收敛性问题。策略梯度具有良好的收敛性证明。
  4. 补充：对于值迭代和策略迭代：策略迭代。它有两个循环，一个是在策略估计的时候，为了求当前策略的值函数需要迭代很多次。另外一个是外面的大循环，就是策略评估，策略提升这个循环。值迭代算法则是一步到位，直接估计最优值函数，因此没有策略提升环节。

- 高冷的面试官：请简述以下时序差分(Temporal Difference，TD)算法。

  答：TD算法是使用广义策略迭代来更新Q函数的方法，核心使用了自举（bootstrapping），即值函数的更新使用了下一个状态的值函数来估计当前状态的值。也就是使用下一步的 $Q$ 值 $Q(S_{t+1},A_{t+1})$ 来更新我这一步的 Q 值 $Q(S_t,A_t) $。完整的计算公式如下：
  $$
  Q(S_t,A_t) \larr Q(S_t,A_t) + \alpha [R_{t+1}+\gamma Q(S_{t+1},A_{t+1})]
  $$
  
- 高冷的面试官：请问蒙特卡洛方法（Monte Carlo Algorithm，MC）和时序差分(Temporal Difference，TD)算法是无偏估计吗？另外谁的方法更大呢？为什么呢？

  答：蒙特卡洛方法（MC）是无偏估计，时序差分（TD）是有偏估计；MC的方差较大，TD的方差较小，原因在于TD中使用了自举（bootstrapping）的方法，实现了基于平滑的效果，导致估计的值函数的方差更小。
  
- 高冷的面试官：能否简单说下动态规划、蒙特卡洛和时序差分的异同点？

  答：

  - 相同点：都用于进行值函数的描述与更新，并且所有方法都是基于对未来事件的展望来计算一个回溯值。

  - 不同点：蒙特卡洛和TD算法隶属于model-free，而动态规划属于model-based；TD算法和蒙特卡洛的方法，因为都是基于model-free的方法，因而对于后续状态的获知也都是基于试验的方法；TD算法和动态规划的策略评估，都能基于当前状态的下一步预测情况来得到对于当前状态的值函数的更新。

    另外，TD算法不需要等到实验结束后才能进行当前状态的值函数的计算与更新，而蒙特卡洛的方法需要试验交互，产生一整条的马尔科夫链并直到最终状态才能进行更新。TD算法和动态规划的策略评估不同之处为model-free和model-based 这一点，动态规划可以凭借已知转移概率就能推断出来后续的状态情况，而TD只能借助试验才能知道。

    蒙特卡洛方法和TD方法的不同在于，蒙特卡洛方法进行完整的采样来获取了长期的回报值，因而在价值估计上会有着更小的偏差，但是也正因为收集了完整的信息，所以价值的方差会更大，原因在于毕竟基于试验的采样得到，和真实的分布还是有差距，不充足的交互导致的较大方差。而TD算法与其相反，因为只考虑了前一步的回报值 其他都是基于之前的估计值，因而相对来说，其估计值具有偏差大方差小的特点。

  - 三者的联系：对于$TD(\lambda)$方法，如果 $ \lambda = 0$ ，那么此时等价于TD，即只考虑下一个状态；如果$ \lambda = 1$，等价于MC，即考虑 $T-1$ 个后续状态即到整个episode序列结束。
