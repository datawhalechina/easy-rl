# Chapter5 Proximal Policy Optimization(PPO) 

## 1 Keywords

- **on-policy(同策略)：** 要learn的agent和环境互动的agent是同一个时，对应的policy。
- **off-policy(异策略)：** 要learn的agent和环境互动的agent不是同一个时，对应的policy。
- **important sampling（重要性采样）：** 使用另外一种数据分布，来逼近所求分布的一种方法，在强化学习中通常和蒙特卡罗方法结合使用，公式如下：$\int f(x) p(x) d x=\int f(x) \frac{p(x)}{q(x)} q(x) d x=E_{x \sim q}[f(x){\frac{p(x)}{q(x)}}]=E_{x \sim p}[f(x)]$  我们在已知 $q$ 的分布后，可以使用上述公式计算出从 $p$ 这个distribution sample x 代入 $f$ 以后所算出来的期望值。
- **Proximal Policy Optimization (PPO)：** 避免在使用important sampling时由于在 $\theta$ 下的 $p_{\theta}\left(a_{t} | s_{t}\right)$ 跟 在  $\theta '$  下的 $p_{\theta'}\left(a_{t} | s_{t}\right)$ 差太多，导致important sampling结果偏差较大而采取的算法。具体来说就是在training的过程中增加一个constrain，这个constrain对应着 $\theta$  跟 $\theta'$  output 的 action 的 KL divergence，来衡量 $\theta$  与 $\theta'$ 的相似程度。

## 2 Questions

- 基于on-policy的policy gradient有什么可改进之处？或者说其效率较低的原因在于？

  答：

  - 经典policy gradient的大部分时间花在sample data处，即当我们的agent与环境做了交互后，我们就要进行policy model的更新。但是对于一个回合我们仅能更新policy model一次，更新完后我们就要花时间去重新collect data，然后才能再次进行如上的更新。

  - 所以我们的可以自然而然地想到，使用off-policy方法使用另一个不同的policy和actor，与环境进行互动并用collect data进行原先的policy的更新。这样等价于使用同一组data，在同一个回合，我们对于整个的policy model更新了多次，这样会更加有效率。

- 使用important sampling时需要注意的问题有哪些。

  答：我们可以在important sampling中将 $p$ 替换为任意的 $q$，但是本质上需要要求两者的分布不能差的太多，即使我们补偿了不同数据分布的权重 $\frac{p(x)}{q(x)}$ 。 $E_{x \sim p}[f(x)]=E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ 当我们对于两者的采样次数都比较多时，最终的结果时一样的，没有影响的。但是通常我们不会取理想的数量的sample data，所以如果两者的分布相差较大，最后结果的variance差距（平方级）将会很大。

- 基于off-policy的importance sampling中的 data 是从 $\theta'$ sample 出来的，从 $\theta$ 换成 $\theta'$ 有什么优势？

  答：使用off-policy的importance sampling后，我们不用 $\theta$ 去跟环境做互动，假设有另外一个 policy  $\theta'$，它就是另外一个actor。它的工作是他要去做demonstration，$\theta'$ 的工作是要去示范给 $\theta$ 看。它去跟环境做互动，告诉 $\theta$ 说，它跟环境做互动会发生什么事。然后，借此来训练$\theta$。我们要训练的是 $\theta$ ，$\theta'$  只是负责做 demo，负责跟环境做互动，所以 sample 出来的东西跟 $\theta$ 本身是没有关系的。所以你就可以让 $\theta'$ 做互动 sample 一大堆的data，$\theta$ 可以update 参数很多次。然后一直到 $\theta$  train 到一定的程度，update 很多次以后，$\theta'$ 再重新去做 sample，这就是 on-policy 换成 off-policy 的妙用。

- 在本节中PPO中的KL divergence指的是什么？

  答：本质来说，KL divergence是一个function，其度量的是两个action （对应的参数分别为$\theta$ 和 $\theta'$ ）间的行为上的差距，而不是参数上的差距。这里行为上的差距（behavior distance）可以理解为在相同的state的情况下，输出的action的差距（他们的概率分布上的差距），这里的概率分布即为KL divergence。 


## 3 Something About Interview

- 高冷的面试官：请问什么是重要性采样呀？

  答：使用另外一种数据分布，来逼近所求分布的一种方法，算是一种期望修正的方法，公式是：
  $$\begin{aligned}
  \int f(x) p(x) d x &= \int f(x) \frac{p(x)}{q(x)} q(x) d x \\
  &= E_{x \sim q}[f(x){\frac{p(x)}{q(x)}}] \\
  &= E_{x \sim p}[f(x)]
  \end{aligned}$$
   我们在已知 $q$ 的分布后，可以使用上述公式计算出从 $p$ 分布的期望值。也就可以使用 $q$ 来对于 $p$ 进行采样了，即为重要性采样。

- 高冷的面试官：请问on-policy跟off-policy的区别是什么？

  答：用一句话概括两者的区别，生成样本的policy（value-funciton）和网络参数更新时的policy（value-funciton）是否相同。具体来说，on-policy：生成样本的policy（value function）跟网络更新参数时使用的policy（value function）相同。SARAS算法就是on-policy的，基于当前的policy直接执行一次action，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同，算法为on-policy算法。该方法会遭遇探索-利用的矛盾，仅利用目前已知的最优选择，可能学不到最优解，收敛到局部最优，而加入探索又降低了学习效率。epsilon-greedy 算法是这种矛盾下的折衷。优点是直接了当，速度快，劣势是不一定找到最优策略。off-policy：生成样本的policy（value function）跟网络更新参数时使用的policy（value function）不同。例如，Q-learning在计算下一状态的预期收益时使用了max操作，直接选择最优动作，而当前policy并不一定能选择到最优动作，因此这里生成样本的policy和学习时的policy不同，即为off-policy算法。

- 高冷的面试官：请简述下PPO算法。其与TRPO算法有何关系呢?

  答：PPO算法的提出：旨在借鉴TRPO算法，使用一阶优化，在采样效率、算法表现，以及实现和调试的复杂度之间取得了新的平衡。这是因为PPO会在每一次迭代中尝试计算新的策略，让损失函数最小化，并且保证每一次新计算出的策略能够和原策略相差不大。具体来说，在避免使用important sampling时由于在 $\theta$ 下的 $p_{\theta}\left(a_{t} | s_{t}\right)$ 跟 在 $\theta'$ 下的 $ p_{\theta'}\left(a_{t} | s_{t}\right) $ 差太多，导致important sampling结果偏差较大而采取的算法。
