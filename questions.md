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

## Chapter2  马尔可夫决策过程（MDP）

#### 1 关键词

- **马尔可夫性质(Markov Property)**：如果某一个过程未来的转移跟过去是独立的，即只取决于现在，那么其满足马尔可夫性质。换句话说，一个状态的下一个状态只取决于它当前状态，而跟它当前状态之前的状态都没有关系。
- **马尔可夫链(Markov Chain)**：概率论和数理统计中具有马尔可夫性质（Markov property）且存在于离散的指数集（index set）和状态空间（state space）内的随机过程（stochastic process）。
- **状态转移矩阵(State Transition Matrix)**：状态转移矩阵类似于一个 conditional probability，当我们知道当前我们在 $s_t$ 这个状态过后，到达下面所有状态的一个概念，它每一行其实描述了是从一个节点到达所有其它节点的概率。
- **马尔可夫奖励过程(Markov Reward Process, MRP)**：即马尔可夫链再加上了一个奖励函数。在 MRP之中，转移矩阵跟它的这个状态都是跟马尔可夫链一样的，多了一个奖励函数(reward function)。奖励函数是一个期望，它说当你到达某一个状态的时候，可以获得多大的奖励。
- **horizon:** 定义了同一个 episode 或者是整个一个轨迹的长度，它是由有限个步数决定的。
- **return:** 把奖励进行折扣(discounted)，然后获得的对应的收益。
- **Bellman Equation（贝尔曼等式）:** 定义了当前状态与未来状态的迭代关系，表示当前状态的值函数可以通过下个状态的值函数来计算。Bellman Equation 因其提出者、动态规划创始人 Richard Bellman 而得名 ，同时也被叫作“动态规划方程”。$V(s)=R(S)+ \gamma \sum_{s' \in S}P(s'|s)V(s')$ ，特别地，矩阵形式：$V=R+\gamma PV$。
- **Monte Carlo Algorithm（蒙特卡罗方法）：** 可用来计算价值函数的值。通俗的讲，我们当得到一个MRP过后，我们可以从某一个状态开始，然后让它让把这个小船放进去，让它随波逐流，这样就会产生一个轨迹。产生了一个轨迹过后，就会得到一个奖励，那么就直接把它的 Discounted 的奖励 $g$ 直接算出来。算出来过后就可以把它积累起来，当积累到一定的轨迹数量过后，然后直接除以这个轨迹，然后就会得到它的这个价值。
- **Iterative Algorithm（动态规划方法）：** 可用来计算价值函数的值。通过一直迭代对应的Bellman Equation，最后使其收敛。当这个最后更新的状态跟你上一个状态变化并不大的时候，这个更新就可以停止。
- **Q函数 (action-value function)：**其定义的是某一个状态某一个行为，对应的它有可能得到的 return 的一个期望（ over policy function）。
- **MDP中的prediction（即policy evaluation问题）：**给定一个 MDP 以及一个 policy $\pi$ ，去计算它的 value function，即每个状态它的价值函数是多少。其可以通过动态规划方法（Iterative Algorithm）解决。
- **MDP中的control问题：** 寻找一个最佳的一个策略，它的 input 就是MDP，输出是通过去寻找它的最佳策略，然后同时输出它的最佳价值函数(optimal value function)以及它的这个最佳策略(optimal policy)。其可以通过动态规划方法（Iterative Algorithm）解决。
- **最佳价值函数(Optimal Value Function)：**我们去搜索一种 policy $\pi$ ，然后我们会得到每个状态它的状态值最大的一个情况，$v^*$ 就是到达每一个状态，它的值的极大化情况。在这种极大化情况上面，我们得到的策略就可以说它是最佳策略(optimal policy)。optimal policy 使得每个状态，它的状态函数都取得最大值。所以当我们说某一个 MDP 的环境被解了过后，就是说我们可以得到一个 optimal value function，然后我们就说它被解了。

#### 2 思考题

- 为什么在马尔可夫奖励过程（MRP）中需要有**discounted factor**?

  答：

  1. 首先，是有些马尔可夫过程是**带环**的，它并没有终结，然后我们想**避免这个无穷的奖励**；
  2. 另外，我们是想把这个**不确定性**也表示出来，希望**尽可能快**地得到奖励，而不是在未来某一个点得到奖励；
  3. 其次，如果这个奖励是它是有实际价值的了，我们可能是更希望立刻就得到奖励，而不是我们后面再得到奖励。
  4. 在人的行为里面来说的话，其实也是大家也是想得到立刻奖励；
  5. 还有在有些时候，这个系数也可以把它设为 0。比如说，当我们设为 0 过后，然后我们就只关注了它当前的奖励。我们也可以把它设为 1，设为 1 的话就是对未来并没有折扣，未来获得的奖励跟我们当前获得的奖励是一样的。

  所以，这个系数其实是应该可以作为强化学习 agent 的一个 hyper parameter 来进行调整，然后就会得到不同行为的 agent。

- 为什么矩阵形式的Bellman Equation的解析解比较难解？

  答：通过矩阵求逆的过程，就可以把这个 V 的这个价值的解析解直接求出来。但是一个问题是这个矩阵求逆的过程的复杂度是 $O(N^3)$。所以就当我们状态非常多的时候，比如说从我们现在十个状态到一千个状态，到一百万个状态。那么当我们有一百万个状态的时候，这个转移矩阵就会是个一百万乘以一百万的一个矩阵。这样一个大矩阵的话求逆是非常困难的，所以这种通过解析解去解，只能对于很小量的MRP。

- 计算贝尔曼等式（Bellman Equation）的常见方法以及区别？

  答：

  1. **Monte Carlo Algorithm（蒙特卡罗方法）：** 可用来计算价值函数的值。通俗的讲，我们当得到一个MRP过后，我们可以从某一个状态开始，然后让它让把这个小船放进去，让它随波逐流，这样就会产生一个轨迹。产生了一个轨迹过后，就会得到一个奖励，那么就直接把它的 Discounted 的奖励 $g$ 直接算出来。算出来过后就可以把它积累起来，当积累到一定的轨迹数量过后，然后直接除以这个轨迹，然后就会得到它的这个价值。
  2. **Iterative Algorithm（动态规划方法）：** 可用来计算价值函数的值。通过一直迭代对应的Bellman Equation，最后使其收敛。当这个最后更新的状态跟你上一个状态变化并不大的时候，这个更新就可以停止。
  3. **以上两者的结合方法：**另外我们也可以通过 Temporal-Difference Learning 的那个办法。这个 `Temporal-Difference Learning` 叫 `TD Leanring`，就是动态规划和蒙特卡罗的一个结合。

- 马尔可夫奖励过程（MRP）与马尔可夫决策过程 （MDP）的区别？

  答：相对于 MRP，马尔可夫决策过程(Markov Decision Process)多了一个 decision，其它的定义跟 MRP 都是类似的。这里我们多了一个决策，多了一个 action ，那么这个状态转移也多了一个 condition，就是采取某一种行为，然后你未来的状态会不同。它不仅是依赖于你当前的状态，也依赖于在当前状态你这个 agent 它采取的这个行为会决定它未来的这个状态走向。对于这个价值函数，它也是多了一个条件，多了一个你当前的这个行为，就是说你当前的状态以及你采取的行为会决定你在当前可能得到的奖励多少。

  另外，两者之间是由转换关系的。具体来说，已知一个 MDP 以及一个 policy $\pi$ 的时候，我们可以把 MDP 转换成MRP。在 MDP 里面，转移函数 $P(s'|s,a)$  是基于它当前状态以及它当前的 action，因为我们现在已知它 policy function，就是说在每一个状态，我们知道它可能采取的行为的概率，那么就可以直接把这个 action 进行加和，那我们就可以得到对于 MRP 的一个转移，这里就没有 action。同样地，对于奖励，我们也可以把 action 拿掉，这样就会得到一个类似于 MRP 的奖励。

- MDP 里面的状态转移跟 MRP 以及 MP 的结构或者计算方面的差异？

  答：

  - 对于之前的马尔可夫链的过程，它的转移是直接就决定，就从你当前是 s，那么就直接通过这个转移概率就直接决定了你下一个状态会是什么。
  - 但是对于 MDP，它的中间多了一层这个行为 a ，就是说在你当前这个状态的时候，你首先要决定的是采取某一种行为。然后因为你有一定的不确定性，当你当前状态决定你当前采取的行为过后，你到未来的状态其实也是一个概率分布。所以你采取行为以及你决定，然后你可能有有多大的概率到达某一个未来状态，以及另外有多大概率到达另外一个状态。所以在这个当前状态跟未来状态转移过程中这里多了一层决策性，这是MDP跟之前的马尔可夫过程很不同的一个地方。在马尔科夫决策过程中，行为是由 agent 决定，所以多了一个 component，agent 会采取行为来决定未来的状态转移。

- 我们如何寻找最佳的policy，方法有哪些？

  答：本质来说，当我们取得最佳的价值函数过后，我们可以通过对这个 Q 函数进行极大化，然后得到最佳的价值。然后，我们直接在这个Q函数上面取一个让这个action最大化的值，然后我们就可以直接提取出它的最佳的policy。

  具体方法：

  1. **穷举法（一般不使用）：**假设我们有有限多个状态、有限多个行为可能性，那么每个状态我们可以采取这个 A 种行为的策略，那么总共就是 $|A|^{|S|}$ 个可能的 policy。我们可以把这个穷举一遍，然后算出每种策略的 value function，然后对比一下可以得到最佳策略。但是效率极低。
  2. **Policy iteration：** 一种迭代方法，有两部分组成，下面两个步骤一直在迭代进行，最终收敛：
     - 第一个步骤是**policy evaluation**，即当前我们在优化这个 policy $\pi$ ，所以在优化过程中得到一个最新的这个 policy 。
     - 第二个步骤是**value iteration**，即取得价值函数后，进一步推算出它的 Q 函数。得到 Q 函数过后，那我们就直接去取它的极大化。
  3. **Value iteration:** 我们一直去迭代 Bellman Optimality Equation，到了最后，它能逐渐趋向于最佳的策略，这是 value iteration 算法的精髓，就是我们去为了得到最佳的 $v^*$ ，对于每个状态它的 $v^*$ 这个值，我们直接把这个 Bellman Optimality Equation 进行迭代，迭代了很多次之后它就会收敛到最佳的policy以及其对应的状态，这里面是没有policy function的。

## Chapter3 表格型方法 

#### 1 关键词

- **P函数和R函数：**P函数就是状态转移的概率，其就是反应的环境的随机性，R函数就是Reward function。但是我们通常处于一个未知的环境（即P函数和R函数是未知的）。
-  **Q表格型表示方法：** 表示形式是一种表格形式，其中横坐标为action（agent）的行为，纵坐标是环境的state，其对应着每一个时刻agent和环境的情况，并通过对应的reward反馈去做选择。一般情况下，Q表格是一个已经训练好的表格，不过，我们也可以每进行一步，就更新一下Q表格，然后用下一个状态的Q值来更新这个状态的Q值（即时序差分方法）。
- **时序差分（Temporal Difference）：** 一种Q函数（Q值）的更新方式，也就是可以拿下一步的 Q 值 $Q(S_{t+_1},A_{t+1})$ 来更新我这一步的 Q 值 $Q(S_t,A_t)$ 。完整的计算公式如下：$Q(S_t,A_t) \larr Q(S_t,A_t) + \alpha [R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$

- **SARSA算法：** 一种更新前一时刻状态的单步更新的强化学习算法，也是一种on-policy策略。该算法由于每次更新值函数需要知道前一步的状态(state)，前一步的动作(action)、奖励(reward)、当前状态(state)、将要执行的动作(action)，即 $(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$ 这几个值，所以被称为SARSA算法。agent没进行一次循环，都会用 $(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$ 对于前一步的Q值（函数）进行一次更新。

#### 2 思考题

- 构成强化学习MDP的四元组？

  答：状态、动作、状态转移概率和奖励，分别对应（S，A，P，R），后面有可能会加上个衰减因子构成五元组。

- 基于以上的描述所构成的强化学习的“学习”流程。

  答：强化学习要像人类一样去学习了，人类学习的话就是一条路一条路的去尝试一下，先走一条路，我看看结果到底是什么。多试几次，只要能一直走下去的，我们其实可以慢慢的了解哪个状态会更好。我们用价值函数 $V(s)$ 来代表这个状态是好的还是坏的。然后用这个 Q 函数来判断说在什么状态下做什么动作能够拿到最大奖励，我们用 Q 函数来表示这个状态-动作值。

- 基于SARSA算法的agent的学习过程。

  答：我们现在有环境，有agent。每交互一次以后，我们的agent会向环境输出action，接着环境会反馈给agent当前时刻的state和reward。那么agent此时会实现两个方法：1. 使用已经训练好的Q表格，对应环境反馈的state和reward选取对应的action并输出。2.我们已经拥有了$(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$  这几个值，并世界使用 $A_{t+1}$ 去更新我们的Q表格。

- Q-learning和Sarsa的区别？

  答：Sarsa算法是Q-learning算法的改进。

  1. 首先，Q-learning 是 off-policy 的时序差分学习方法，Sarsa 是 on-policy 的时序差分学习方法。

  2. 其次，Sarsa 在更新 Q 表格的时候，它用到的 A' 。我要获取下一个 Q 值的时候，A' 是下一个 step 一定会执行的 action 。这个 action 有可能是 $\varepsilon$-greddy 方法 sample 出来的值，也有可能是 max Q 对应的 action，也有可能是随机动作。但是就是它实实在在执行了的那个动作。

  3. 但是 Q-learning 在更新 Q 表格的时候，它用到这个的 Q 值 $Q(S',a')$ 对应的那个 action ，它不一定是下一个 step 会执行的实际的 action，因为你下一个实际会执行的那个 action 可能会探索。Q-learning 默认的 action 不是通过 behavior policy 来选取的，它是默认 A' 为最优策略选的动作，所以 Q-learning 在学习的时候，不需要传入 A'，即 $a_{t+1}$  的值。

  4. 更新公式的对比（区别只在target计算这一部分）：

     - Sarsa的公式： $R_{t+1}+\gamma Q(S_{t+1}, A_{t+1})$ ；
     - Q-learning的公式：$R_{t+1}+\gamma  \underset{a}{\max} Q\left(S_{t+1}, a\right)$

     Sarsa 实际上都是用自己的策略产生了 S,A,R,S',A' 这一条轨迹。然后拿着 $Q(S_{t+1},A_{t+1})$ 去更新原本的 Q 值 $Q(S_t,A_t)$。 但是 Q-learning 并不需要知道，我实际上选择哪一个 action ，它默认下一个动作就是 Q 最大的那个动作。

- On-policy和 off-policy 的区别？

  答：

  1. Sarsa 就是一个典型的 on-policy 策略，它只用一个 $\pi$ ，为了兼顾探索和利用，所以它训练的时候会显得有点胆小怕事。它在解决悬崖问题的时候，会尽可能地离悬崖边上远远的，确保说哪怕自己不小心探索了一点了，也还是在安全区域内不不至于跳进悬崖。
  2. Q-learning 是一个比较典型的 off-policy 的策略，它有目标策略 target policy，一般用 $\pi$ 来表示。然后还有行为策略 behavior policy，用 $\mu$ 来表示。它分离了目标策略跟行为策略。Q-learning 就可以大胆地用 behavior policy 去探索得到的经验轨迹来去优化我的目标策略。这样子我更有可能去探索到最优的策略。
  3. 比较 Q-learning 和 Sarsa 的更新公式可以发现，Sarsa 并没有选取最大值的 max 操作。因此，Q-learning 是一个非常激进的算法，希望每一步都获得最大的利益；而 Sarsa 则相对非常保守，会选择一条相对安全的迭代路线。

## Chapter4 梯度策略 

#### 1 关键词

- **policy（策略）：** 每一个actor中会有对应的策略，这个策略决定了actor的行为。具体来说，Policy 就是给一个外界的输入，然后它会输出 actor 现在应该要执行的行为。**一般地，我们将policy写成 $\pi$ 。**
- **Return（回报）：** 一个回合（Episode）或者试验（Trial）所得到的所有的reward的总和，也被人们称为Total reward。**一般地，我们用 $R$ 来表示他。**
- **Trajectory：** 一个试验中我们将environment 输出的 $s$ 跟 actor 输出的行为 $a$，把这个 $s$ 跟 $a$ 全部串起来形成的集合，我们称为Trajectory，即 $\text { Trajectory } \tau=\left\{s_{1}, a_{1}, s_{2}, a_{2}, \cdots, s_{t}, a_{t}\right\}$。
- **Reward function：** 根据在某一个 state 采取的某一个 action 决定说现在这个行为可以得到多少的分数，它是一个 function。也就是给一个 $s_1$，$a_1$，它告诉你得到 $r_1$。给它 $s_2$ ，$a_2$，它告诉你得到 $r_2$。 把所有的 $r$ 都加起来，我们就得到了 $R(\tau)$ ，代表某一个 trajectory $\tau$ 的 reward。
- **Expected reward：** $\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]$。
- **Reinforce：** 基于策略梯度的强化学习的经典算法，其采用回合更新的模式。

#### 2 思考题

- 如果我们想让机器人自己玩video game, 那么强化学习中三个组成（actor、environment、reward function）部分具体分别是什么？

  答：actor 做的事情就是去操控游戏的摇杆， 比如说向左、向右、开火等操作；environment 就是游戏的主机， 负责控制游戏的画面负责控制说，怪物要怎么移动， 你现在要看到什么画面等等；reward function 就是当你做什么事情，发生什么状况的时候，你可以得到多少分数， 比如说杀一只怪兽得到 20 分等等。

- 在一个process中，一个具体的trajectory $s_1$,$a_1$, $s_2$ , $a_2$ 出现的概率取决于什么？

  答：

  1. 一部分是 **environment 的行为**， environment 的 function 它内部的参数或内部的规则长什么样子。 $p(s_{t+1}|s_t,a_t)$这一项代表的是 environment， environment 这一项通常你是无法控制它的，因为那个是人家写好的，你不能控制它。

  2. 另一部分是 **agent 的行为**，你能控制的是 $p_\theta(a_t|s_t)$。给定一个 $s_t$， actor 要采取什么样的 $a_t$ 会取决于你 actor 的参数 $\theta$， 所以这部分是 actor 可以自己控制的。随着 actor 的行为不同，每个同样的 trajectory， 它就会有不同的出现的概率。

- 当我们在计算 maximize expected reward时，应该使用什么方法？

  答： **gradient ascent（梯度上升）**，因为要让它越大越好，所以是 gradient ascent。Gradient ascent 在 update 参数的时候要加。要进行 gradient ascent，我们先要计算 expected reward $\bar{R}$ 的 gradient 。我们对 $\bar{R}$ 取一个 gradient，这里面只有 $p_{\theta}(\tau)$ 是跟 $\theta$ 有关，所以 gradient 就放在 $p_{\theta}(\tau)$ 这个地方。

- 我们应该如何理解梯度策略的公式呢？

  答：
  $$
  \begin{aligned}
  E_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] &\approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(\tau^{n}\right) \\
  &=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)
  \end{aligned}
  $$
   $p_{\theta}(\tau)$ 里面有两项，$p(s_{t+1}|s_t,a_t)$ 来自于 environment，$p_\theta(a_t|s_t)$ 是来自于 agent。 $p(s_{t+1}|s_t,a_t)$ 由环境决定从而与 $\theta$ 无关，因此 $\nabla \log p(s_{t+1}|s_t,a_t) =0 $。因此 $\nabla p_{\theta}(\tau)=
  \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)$。

  具体来说：

  *  假设你在 $s_t$ 执行 $a_t$，最后发现 $\tau$ 的 reward 是正的， 那你就要增加这一项的概率，你就要增加在 $s_t$ 执行 $a_t$ 的概率。
  *  反之，在 $s_t$ 执行 $a_t$ 会导致$\tau$  的 reward 变成负的， 你就要减少这一项的概率。

- 我们可以使用哪些方法来进行gradient ascent的计算？

  答：用 gradient ascent 来 update 参数，对于原来的参数 $\theta$ ，可以将原始的 $\theta$  加上更新的 gradient 这一项，再乘以一个 learning rate，learning rate 其实也是要调的，和神经网络一样，我们可以使用 Adam、RMSProp 等优化器对其进行调整。

- 我们进行基于梯度策略的优化时的小技巧有哪些？

  答：

  1. **Add a baseline：**为了防止所有的reward都大于0，从而导致每一个stage和action的变换，会使得每一项的概率都会上升。所以通常为了解决这个问题，我们把reward 减掉一项叫做 b，这项 b 叫做 baseline。你减掉这项 b 以后，就可以让 $R(\tau^n)-b$ 这一项， 有正有负。 所以如果得到的 total reward $R(\tau^n)$ 大于 b 的话，就让它的概率上升。如果这个 total reward 小于 b，就算它是正的，正的很小也是不好的，你就要让这一项的概率下降。 如果$R(\tau^n)<b$  ， 你就要让这个 state 采取这个 action 的分数下降 。这样也符合常理。
  2. **Assign suitable credit：** 首先第一层，本来的 weight 是整场游戏的 reward 的总和。那现在改成从某个时间 $t$ 开始，假设这个 action 是在 t 这个时间点所执行的，从 $t$ 这个时间点，一直到游戏结束所有 reward 的总和，才真的代表这个 action 是好的还是不好的；接下来我们再进一步，我们把未来的reward做一个discount，这里我们称由此得到的reward的和为**Discounted Return(折扣回报)** 。
  3. 综合以上两种tip，我们将其统称为**Advantage function**， 用 `A` 来代表 advantage function。Advantage function 是 dependent on s and a，我们就是要计算的是在某一个 state s 采取某一个 action a 的时候，advantage function 有多大。
  4. Advantage function 的意义就是，假设我们在某一个 state $s_t$ 执行某一个 action $a_t$，相较于其他可能的 action，它有多好。它在意的不是一个绝对的好，而是相对的好，即相对优势(relative advantage)。因为会减掉一个 b，减掉一个 baseline， 所以这个东西是相对的好，不是绝对的好。 $A^{\theta}\left(s_{t}, a_{t}\right)$ 通常可以是由一个 network estimate 出来的，这个 network 叫做 critic。
  
- 对于梯度策略的两种方法，蒙特卡洛（MC）强化学习和时序差分（TD）强化学习两个方法有什么联系和区别？

  答：

  1. **两者的更新频率不同**，蒙特卡洛强化学习方法是**每一个episode更新一次**，即需要经历完整的状态序列后再更新，而对于时序差分强化学习方法是**每一个step就更新一次**。相对来说，时序差分强化学习方法比蒙特卡洛强化学习方法更新的频率更快。
  2. 时序差分强化学习能够在知道一个小step后就进行学习，相比于蒙特卡洛强化学习，其更加**快速、灵活**。
  3. 具体举例来说：假如我们要优化开车去公司的通勤时间。对于此问题，每一次通勤，我们将会到达不同的路口。对于时序差分（TD）强化学习，其会对于每一个经过的路口都会计算时间，例如在路口 A 就开始更新预计到达路口 B、路口 C $\cdots \cdots$, 以及到达公司的时间；而对于蒙特卡洛（MC）强化学习，其不会每经过一个路口就更新时间，而是到达最终的目的地后，再修改每一个路口和公司对应的时间。

- 请详细描述REINFORCE的计算过程。

  答：首先我们需要根据一个确定好的policy model来输出每一个可能的action的概率，对于所有的action的概率，我们使用sample方法（或者是随机的方法）去选择一个action与环境进行交互，同时环境就会给我们反馈一整个episode数据。对于此episode数据输入到learn函数中，并根据episode数据进行loss function的构造，通过adam等优化器的优化，再来更新我们的policy model。

## Chapter5 Proximal Policy Optimization(PPO) 

#### 1 关键词

- **on-policy(同策略)：** 要learn的agent和环境互动的agent是同一个时，对应的policy。
- **off-policy(异策略)：** 要learn的agent和环境互动的agent不是同一个时，对应的policy。
- **important sampling（重要性采样）：** 使用另外一种数据分布，来逼近所求分布的一种方法，在强化学习中通常和蒙特卡罗方法结合使用，公式如下：$\int f(x) p(x) d x=\int f(x) \frac{p(x)}{q(x)} q(x) d x=E_{x \sim q}[f(x){\frac{p(x)}{q(x)}}]=E_{x \sim p}[f(x)]$  我们在已知 $q$ 的分布后，可以使用上述公式计算出从 $p$ 这个distribution sample x 代入 $f$ 以后所算出来的期望值。
- **Proximal Policy Optimization (PPO)：** 避免在使用important sampling时由于在 $\theta$ 下的 $p_{\theta}\left(a_{t} | s_{t}\right)$ 跟 在  $\theta '$  下的 $p_{\theta'}\left(a_{t} | s_{t}\right)$ 差太多，导致important sampling结果偏差较大而采取的算法。具体来说就是在training的过程中增加一个constrain，这个constrain对应着 $\theta$  跟 $\theta'$  output 的 action 的 KL divergence，来衡量 $\theta$  与 $\theta'$ 的相似程度。

#### 2 思考题

- 基于on-policy的policy gradient有什么可改进之处？或者说其效率较低的原因在于？

  答：

  - 经典policy gradient的大部分时间花在sample data处，即当我们的agent与环境做了交互后，我们就要进行policy model的更新。但是对于一个回合我们仅能更新policy model一次，更新完后我们就要花时间去重新collect data，然后才能再次进行如上的更新。

  - 所以我们的可以自然而然地想到，使用off-policy方法使用另一个不同的policy和actor，与环境进行互动并用collect data进行原先的policy的更新。这样等价于使用同一组data，在同一个回合，我们对于整个的policy model更新了多次，这样会更加有效率。

- 使用important sampling时需要注意的问题有哪些。

  答：我们可以在important sampling中将 $p$ 替换为任意的 $q$，但是本质上需要要求两者的分布不能差的太多，即使我们补偿了不同数据分布的权重 $\frac{p(x)}{q(x)}$ 。 $E_{x \sim p}[f(x)]=E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ 当我们对于两者的采样次数都比较多时，最终的结果时一样的，没有影响的。但是通常我们不会取理想的数量的data，所以如果两者的分布相差较大，最后结果的variance差距将会很大。

- 基于off-policy的importance sampling中的 data 是从 $\theta'$ sample 出来的，从 $\theta$ 换成 $\theta'$ 有什么优势？

  答：使用off-policy的importance sampling后，我们不用$\theta$ 去跟环境做互动，假设有另外一个 policy  $\theta'$，它就是另外一个actor。它的工作是他要去做demonstration，$\theta'$ 的工作是要去示范给$\theta$ 看。它去跟环境做互动，告诉 $\theta$ 说，它跟环境做互动会发生什么事。然后，借此来训练$\theta$。我们要训练的是 $\theta$ ，$\theta'$  只是负责做 demo，负责跟环境做互动，所以 sample 出来的东西跟 $\theta$ 本身是没有关系的。所以你就可以让 $\theta'$ 做互动 sample 一大堆的data，$\theta$ 可以update 参数很多次。然后一直到 $\theta$  train 到一定的程度，update 很多次以后，$\theta'$ 再重新去做 sample，这就是 on-policy 换成 off-policy 的妙用。

- 在本节中PPO中的KL divergence指的是什么？

  答：本质来说，KL divergence是一个function，其度量的是两个action （对应的参数分别为$\theta$ 和 $\theta'$ ）间的行为上的差距，而不是参数上的差距。这里行为上的差距（behavior distance）可以理解为在相同的state的情况下，输出的action的差距（他们的概率分布上的差距），这里的概率分布即为KL divergence。

## Chapter6 Q-learning-State Value Function

#### 1 关键词

- **DQN(Deep Q-Network)：**  基于深度学习的Q-learning算法，其结合了 Value Function Approximation（价值函数近似）与神经网络技术，并采用了目标网络（Target Network）和经历回放（Experience Replay）的方法进行网络的训练。
- **State-value Function：** 本质是一种critic。其输入为actor某一时刻的state，对应的输出为一个标量，即当actor在对应的state时，预期的到过程结束时间段中获得的value的数值。
- **State-value Function Bellman Equation：** 基于state-value function的Bellman Equation，它表示在状态 $s_t$ 下带来的累积奖励 $G_t$ 的期望。
- **Q-function:** 其也被称为state-action value function。其input 是一个 state 跟 action 的 pair，即在某一个 state 采取某一个action，假设我们都使用 actor $\pi$ ，得到的 accumulated reward 的期望值有多大。
- **Target Network：** 为了解决在基于TD的Network的问题时，优化目标 $\mathrm{Q}^{\pi}\left(s_{t}, a_{t}\right) 
  =r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 左右两侧会同时变化使得训练过程不稳定，从而增大regression的难度。target network选择将上式的右部分即 $r_{t}+\mathrm{Q}^{\pi}\left(s_{t+1}, \pi\left(s_{t+1}\right)\right)$ 固定，通过改变上式左部分的network的参数，进行regression。也是一个DQN中比较重要的tip。
- **Exploration：**  在我们使用Q-function的时候，我们的policy完全取决于Q-function，有可能导致出现对应的action是固定的某几个数值的情况，而不像policy gradient中的output为随机的，我们再从随机的distribution中sample选择action。这样会导致我们继续训练的input的值一样，从而”加重“output的固定性，导致整个模型的表达能力的急剧下降，这也就是`探索-利用窘境(Exploration-Exploitation dilemma)`问题。所以我们使用`Epsilon Greedy`和 `Boltzmann Exploration`等Exploration方法进行优化。
- **Experience Replay（经验回放）：**  其会构建一个Replay Buffer（Replay Memory），用来保存许多data，每一个data的形式如下：在某一个 state $s_t$，采取某一个action $a_t$，得到了 reward $r_t$，然后跳到 state $s_{t+1}$。我们使用 $\pi$ 去跟环境互动很多次，把收集到的数据都放到这个 replay buffer 中。当我们的buffer”装满“后，就会自动删去最早进入buffer的data。在训练时，对于每一轮迭代都有相对应的batch（与我们训练普通的Network一样通过sample得到），然后用这个batch中的data去update我们的Q-function。综上，Q-function再sample和训练的时候，会用到过去的经验数据，所以这里称这个方法为Experience Replay，其也是DQN中比较重要的tip。

#### 2 思考题

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

## Chapter7 Q-learning-Double DQN

#### 1 关键词

- **Double DQN：** 在Double DQN中存在有两个 Q-network，首先，第一个 Q-network，决定的是哪一个 action 的 Q value 最大，从而决定了你的action。另一方面， Q value 是用 $Q'$ 算出来的，这样就可以避免 over estimate 的问题。具体来说，假设我们有两个 Q-function，假设第一个Q-function 它高估了它现在选出来的action a，那没关系，只要第二个Q-function $Q'$ 没有高估这个action a 的值，那你算出来的，就还是正常的值。
- **Dueling DQN：** 将原来的DQN的计算过程分为**两个path**。对于第一个path，会计算一个于input state有关的一个标量 $V(s)$；对于第二个path，会计算出一个vector $A(s,a)$ ，其对应每一个action。最后的网络是将两个path的结果相加，得到我们最终需要的Q value。用一个公式表示也就是 $Q(s,a)=V(s)+A(s,a)$ 。 
- **Prioritized Experience Replay （优先经验回放）：** 这个方法是为了解决我们在chapter6中提出的**Experience Replay（经验回放）**方法不足进一步优化提出的。我们在使用Experience Replay时是uniformly取出的experience buffer中的sample data，这里并没有考虑数据间的权重大小。例如，我们应该将那些train的效果不好的data对应的权重加大，即其应该有更大的概率被sample到。综上， prioritized experience replay 不仅改变了 sample data 的 distribution，还改变了 training process。
- **Noisy Net：** 其在每一个episode 开始的时候，即要和环境互动的时候，将原来的Q-function 的每一个参数上面加上一个Gaussian noise。那你就把原来的Q-function 变成$\tilde{Q}$ ，即**Noisy Q-function**。同样的我们把每一个network的权重等参数都加上一个Gaussian noise，就得到一个新的network $\tilde{Q}$。我们会使用这个新的network从与环境互动开始到互动结束。
- **Distributional Q-function：** 对于DQN进行model distribution。将最终的网络的output的每一类别的action再进行distribution。
- **Rainbow：** 也就是将我们这两节内容所有的七个tips综合起来的方法，7个方法分别包括：DQN、DDQN、Prioritized DDQN、Dueling DDQN、A3C、Distributional DQN、Noisy DQN，进而考察每一个方法的贡献度或者是否对于与环境的交互式正反馈的。

#### 2 思考题

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

## Chapter8 Q-learning for Continuous Actions

#### 思考题

- Q-learning相比于policy gradient based方法为什么训练起来效果更好，更平稳？

  答：在 Q-learning 中，只要能够 estimate 出Q-function，就可以保证找到一个比较好的 policy，同样的只要能够 estimate 出 Q-function，就保证可以 improve 对应的 policy。而因为 estimate Q-function 作为一个回归问题，是比较容易的。在这个回归问题中， 我们可以时刻观察我们的模型训练的效果是不是越来越好，一般情况下我们只需要关注 regression 的 loss 有没有下降，你就知道你的 model learn 的好不好。所以 estimate Q-function 相较于 learn 一个 policy 是比较容易的。你只要 estimate Q-function，就可以保证说现在一定会得到比较好的 policy，同样其也比较容易操作。

- Q-learning在处理continuous action时存在什么样的问题呢？

  答：在日常的问题中，我们的问题都是continuous action的，例如我们的 agent 要做的事情是开自驾车，它要决定说它方向盘要左转几度， 右转几度，这就是 continuous 的；假设我们的 agent 是一个机器人，假设它身上有 50 个关节，它的每一个 action 就对应到它身上的这 50 个关节的角度，而那些角度也是 continuous 的。

  然而在解决Q-learning问题时，很重要的一步是要求能够解对应的优化问题。当我们 estimate 出Q-function $Q(s,a)$ 以后,必须要找到一个 action，它可以让 $Q(s,a)$ 最大。假设 action 是 discrete 的，那 a 的可能性都是有限的。但如果action是continuous的情况下，我们就不能像离散的action一样，穷举所有可能的continuous action了。

  为了解决这个问题，有以下几种solutions：

  - 第一个解决方法：我们可以使用所谓的sample方法，即随机sample出N个可能的action，然后一个一个带到我们的Q-function中，计算对应的N个Q value比较哪一个的值最大。但是这个方法因为是sample所以不会非常的精确。
  - 第二个解决方法：我们将这个continuous action问题，看为一个优化问题，从而自然而然地想到了可以用gradient ascend去最大化我们的目标函数。具体地，我们将action看为我们的变量，使用gradient ascend方法去update action对应的Q-value。但是这个方法通常的时间花销比较大，因为是需要迭代运算的。
  - 第三个解决方法：设计一个特别的network架构，设计一个特别的Q-function，使得解我们 argmax Q-value的问题变得非常容易。也就是这边的 Q-function 不是一个 general 的 Q-function，特别设计一下它的样子，让你要找让这个 Q-function 最大的 a 的时候非常容易。但是这个方法的function不能随意乱设，其必须有一些额外的限制。具体的设计方法，可以我们的chapter8的详细教程。
  - 第四个解决方法：不用Q-learning，毕竟用其处理continuous的action比较麻烦。

## Chapter9 Actor-Critic

#### 1 关键词

- **A2C：** Advantage Actor-Critic的缩写，一种Actor-Critic方法。

- **A3C：** Asynchronous（异步的）Advantage Actor-Critic的缩写，一种改进的Actor-Critic方法，通过异步的操作，进行RL模型训练的加速。
-  **Pathwise Derivative Policy Gradient：** 其为使用 Q-learning 解 continuous action 的方法，也是一种 Actor-Critic 方法。其会对于actor提供value最大的action，而不仅仅是提供某一个action的好坏程度。

#### 2 思考题

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

## Chapter10 Sparse Reward

#### 1 关键词

- **reward shaping：** 在我们的agent与environment进行交互时，我们人为的设计一些reward，从而“指挥”agent，告诉其采取哪一个action是最优的，而这个reward并不是environment对应的reward，这样可以提高我们estimate Q-function时的准确性。
- **ICM（intrinsic curiosity module）：** 其代表着curiosity driven这个技术中的增加新的reward function以后的reward function。
- **curriculum learning：** 一种广义的用在RL的训练agent的方法，其在input训练数据的时候，采取由易到难的顺序进行input，也就是认为设计它的学习过程，这个方法在ML和DL中都会普遍使用。
- **reverse curriculum learning：** 相较于上面的curriculum learning，其为更general的方法。其从最终最理想的state（我们称之为gold state）开始，依次去寻找距离gold state最近的state作为想让agent达到的阶段性的“理想”的state，当然我们应该在此过程中有意的去掉一些极端的case（太简单、太难的case）。综上，reverse curriculum learning 是从 gold state 去反推，就是说你原来的目标是长这个样子，我们从我们的目标去反推，所以这个叫做 reverse curriculum learning。  
- **hierarchical （分层） reinforcement learning：** 将一个大型的task，横向或者纵向的拆解成多个 agent去执行。其中，有一些agent 负责比较high level 的东西，负责订目标，然后它订完目标以后，再分配给其他的 agent把它执行完成。（看教程的 hierarchical  reinforcement learning部分的示例就会比较明了）

#### 2 思考题

- 解决sparse reward的方法有哪些？

  答：Reward Shaping、curiosity driven reward、（reverse）curriculum learning 、Hierarchical Reinforcement learning等等。

- reward shaping方法存在什么主要问题？

  答：主要的一个问题是我们人为设计的reward需要domain knowledge，需要我们自己设计出符合environment与agent更好的交互的reward，这需要不少的经验知识，需要我们根据实际的效果进行调整。

- ICM是什么？我们应该如何设计这个ICM？

  答：ICM全称为intrinsic curiosity module。其代表着curiosity driven这个技术中的增加新的reward function以后的reward function。具体来说，ICM在更新计算时会考虑三个新的东西，分别是 state $s_1$、action $a_1$ 和 state $s_2$。根据$s_1$ 、$a_1$、 $a_2$，它会 output 另外一个新的 reward $r_1^i$。所以在ICM中我们total reward 并不是只有 r 而已，还有 $r^i$。它不是只有把所有的 r 都加起来，它还把所有 $r^i$ 加起来当作total reward。所以，它在跟环境互动的时候，它不是只希望 r 越大越好，它还同时希望 $r^i$ 越大越好，它希望从 ICM 的 module 里面得到的 reward 越大越好。ICM 就代表了一种curiosity。

  对于如何设计ICM，ICM的input就像前面所说的一样包括三部分input 现在的 state $s_1$，input 在这个 state 采取的 action $a_1$，然后接 input 下一个 state $s_{t+1}$，对应的output就是reward $r_1^i$，input到output的映射是通过network构建的，其使用 $s_1$ 和 $a_1$ 去预测 $\hat{s}_{t+1}$ ,然后继续评判预测的$\hat{s}_{t+1}$和真实的$s_{t+1}$像不像，越不相同得到的reward就越大。通俗来说这个reward就是，如果未来的状态越难被预测的话，那么得到的reward就越大。这也就是curiosity的机制，倾向于让agent做一些风险比较大的action，从而增加其machine exploration的能力。

  同时为了进一步增强network的表达能力，我们通常讲ICM的input优化为feature extractor，这个feature extractor模型的input就是state，output是一个特征向量，其可以表示这个state最主要、重要的特征，把没有意义的东西过滤掉。