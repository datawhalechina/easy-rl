## Asynchronous Methods for Deep Reinforcement Learning（深度强化学习的异步方法）

作者：Volodymyr Mnih，Adrià Puigdomènech Badia，Mehdi Mirza，Alex Graves，Tim Harley，Timothy P. Lillicrap，David Silver，Koray Kavukcuoglu

单位：Google DeepMind，Montreal Institute for Learning Algorithms （MILA）, University of Montreal

会议：International Conference on Machine Learning（ICML 2016）

论文地址：https://arxiv.org/abs/1602.01783

论文贡献：提出一种异步梯度优化框架，在各类型强化学习算法上都有较好效果，其中效果最好的就是异步优势演员-评论员（Asynchronous advantage actor-critic ，A3C）。

- **Motivation(Why)**：神经网络与强化学习算法相结合可以让算法更有效，但稳定性较差。通常的做法是使用经验回放来去除数据的相关性，让训练更加稳定。但这个方法有两个缺点：一是需要较多的内存和算力，二是只适用于异策略（off-policy）强化学习算法。
- **Main Idea(What)**：本文提出使用异步并行的方法来代替经验回放技术，这种方法不仅可以去除数据相关性，所需资源较少，并且可以应用于多种类型强化学习方法。



### Asynchronous RL Framework（异步强化学习架构）

本文将异步架构用于四种强化学习算法上：一步Sarsa，一步Q学习，$n$ 步Q学习以及优势演员-评论员（advantage actor-critic ，A2C）。这里可以看到此架构对于同/异策略、基于价值/策略的强化学习算法均适用。

异步框架有两大特点：

1. 只使用一台带有多核CPU的机器，不需要GPU。这种方式可以减少机器间的通信成本，并且可以用Hogwild! 的方式进行训练。
2. 多个行动者（actor/worker）并行可以采取不同的策略与环境互动让数据关联性减弱从而不需要经验回放技术，这使得框架可以用于同策略方法，并且训练时间与行动者数量近似线性关系。

先来聊聊Hogwild! ，这是框架的核心，不清楚的可参考[知乎讲解](https://zhuanlan.zhihu.com/p/30826161)，下图来源此文：

![image-20221109221145358](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20221109221145358.png)

这里举个栗子：假设网络有100个参数，worker $i$传来的梯度更新了40个参数后worker $j$传来的就开始更新了，当$i$更新完后前面的一些参数被$j$被覆盖掉了，但不要紧，$i$更新完了照样把当前更新后参数同步给$i$，这就是异步的意思。

接下来介绍四种异步算法：

#### Asynchronous one-step Q-learning（异步一步Q学习）

![image-20230113230006360](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230113230006360.png)

- $\theta$ 和 $\theta^-$是共享的，任何线程可以修改
- 梯度累积是一个增加batch size的技巧，并且可以减少覆盖其他线程更新的机会（更新频率降低了）
- $I_{target}$是同步两网络参数的间隔
- $I_{AsyncUpdate}$是线程更新共享参数$\theta$的间隔

#### Asynchronous one-step Sarsa（异步一步Sarsa）

与上面的异步一步Q学习相比，只有目标价值（target value）变为了$r+\gamma Q(s^\prime,a^\prime;\theta^\prime)$，其余部分相同。

#### Asynchronous n-step Q-learning（异步n步Q学习）

![image-20230113231647918](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230113231647918.png)

- 一步法的缺点是 $r$ 只直接影响导致$r$产生的 $Q(s,a)$ 值，其他的$Q$值被间接影响。$n$步法得到的$r$可以影响前面$n$步的$Q$值，这使得通过$r$更新$Q$值更加有效。后文实验有关于两者的比较。
- 相较与一步法多了个线程专属参数$\theta^{\prime}$，这个参数在选取动作以及计算梯度的时候会使用，不受其他线程的影响，可以让训练更稳定。

#### Asynchronous advantage actor-critic（A3C）

![image-20221109234156910](https://gitee.com/xyfcw/CloudPictures/raw/master/img/202211092341970.png)

- 共享了演员，评论家两网络参数，每个线程也有自己专属的，其他的与$n$步Q学习基本一样



### 实验

#### Atari游戏

在5个Atari游戏上比较算法训练速度：

![image-20221110000219965](https://gitee.com/xyfcw/CloudPictures/raw/master/img/202211100002021.png)

- DQN用K40GPU，异步方法用16核的CPU，可以看到异步优于DQN，并且$n$步方法是优于一步方法的，A3C明显优于其他算法。

在57个Atari游戏上比较算法性能：

![image-20230114134153685](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114134153685.png)

- 把A3C与当时在Atari游戏上最好的一些算法进行比较，可以看到A3C训练4天的性能强于其他算法训练8天的水平。

#### TORCS Car Racing Simulator（TORCS赛车模拟器）

在比Atari游戏更难的赛车环境里检验异步算法性能：

![image-20230114134919316](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114134919316.png)

- 可以看到A3C也是明显优于其他算法。

#### Continuous Action Control Using the MuJoCo Physics Simulator（使用MuJoCo物理模拟器进行连续动作控制）

在Mujoco环境里检验A3C对于连续型动作状态任务的性能：

![image-20230114145118141](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114145118141.png)

- 在绝大多数任务上A3C在学习率（learning rate）在大范围变动下都有很好的表现。

#### Labyrinth（迷宫）

这是一个3D迷宫环境，每一轮（episode）都会随机生成一个迷宫，迷宫里有苹果和门，智能体找到一个苹果得1分，找到门得10分。找到门后智能体会被随机放到迷宫任意一个地方，并且之前被找到的苹果会重新生成。一轮持续60秒，结束后新的一轮重新开始。

这个环境需要智能体找到一种通用的策略，因为每一轮迷宫都不一样，这对智能体挑战较大，但作者训练A3C后发现此算法能达到很好的效果。

#### Scalability and Data Efficiency（可扩展性和数据效率）

在Atari游戏上检验算法的训练时间和数据效率随着线程数变化会如何改变：

![image-20230114151851212](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114151851212.png)

- 作者提出用一个线程达到固定分数的时间除以$n$个线程达到分数的时间，来衡量算法通过增加线程的提速程度。结果表明训练速度会随着线程数的增加而提升，这证明异步框架可以很好地利用训练资源。

#### Robustness and Stability（稳健性和稳定性）

在5个Atari游戏中检验异步算法的稳健性和稳定性：

A3C:

![image-20230114164256003](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114164256003.png)

其他的三种异步算法：

![image-20230114164313044](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20230114164313044.png)

- 结果表明所有的算法在学习率在大范围变化时都有较好的稳健性和稳定性。



### 总结

异步框架有几大优点：

- 效果好的同时所需资源较少
- 可以应用于同策略、异策略等多种强化学习算法
- 数据利用率高，稳定性较好
- 可以广泛结合其他算法的优点提升性能



### 参考资料

- [Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)

- [知乎讲解](https://zhuanlan.zhihu.com/p/310608740)
- [知乎讲解](https://zhuanlan.zhihu.com/p/136823256)



作者：汪聪

单位：武汉工程大学

研究方向：机器博弈，强化学习

联系方式：xyfcw@qq.com