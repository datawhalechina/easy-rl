
最近 Genie3 很火，但大家对世界模型的概念还有些模糊，有人认为生成模型就是世界模型，但让我们回顾最早的世界模型定义，也就是 Genie1 论文“Genie: Generative Interactive Environments”中提到的定义：”生成式交互环境可以被视为世界模型的一类，它们能够在给定动作输入的条件下，实现下一帧的预测“，由此可知，生成模型≠世界模型。接下来给大家介绍下做强化学习的人眼中的世界模型。

<div align=center> <img width="550" src="../img/ch16/image 1.png"/> </div> <div align=center></div>

## 为什么是世界模型？

**使用世界模型可以替代环境训练，我们可以完全在模拟的“梦境”环境中训练智能体，而不需要与真实环境交互**。这种方法提供了诸多好处。例如，运行计算密集型的游戏引擎需要使用大量的计算资源来将游戏状态渲染成图像帧，或者计算与游戏不直接相关的物理内容。我们可能不想在实际环境中浪费计算资源来训练智能体，而是可以在其模拟环境中多次训练智能体。现实世界中训练智能体的成本更高，因此逐步训练的世界模型用于模拟现实可能有助于将策略转移到现实世界中。

<div align=center> <img width="550" src="../img/ch16/image 2.png"/> </div> <div align=center></div>

## 世界模型是什么？

最早的世界模型概念来自 David Ha 和 LSTM 之父  Jürgen Schmidhuber 的 NIPS 2018 Oral Presentation 的论文 “World Models”。这篇论文给出的世界模型由以下3个部分组成：

- **V 模型（Variational Autoencoder，VAE）**：这是智能体（agent）的视觉感知部分，用于将高维的图像帧（如来自游戏环境的2D图像）压缩成低维的 latent 表示。这个模型对于智能体从原始输入数据中学习有意义的、抽象的表示至关重要。

<div align=center> <img width="550" src="../img/ch16/image 3.png"/> </div> <div align=center></div>

- **M 模型（MDN-RNN）**的作用是预测未来，具体来说，M 模型根据当前时刻 $t$ 的隐向量（latent vector） $$z_t$$ 、隐状态（hidden state） $h_t$ 以及动作$a_t$来预测下一时刻的隐向量 $z_{t+1}$ 。它使用**混合密度网络（Mixture Density Network，MDN）**与**循环神经网络（Recurrent Neural Network，RNN）**结合的方式，输出下一个隐向量 $z$ 的概率分布。温度参数 $\tau$ 用来控制模型的不确定性。

$$
P\left(z_{t+1} \mid a_t, z_t, h_t\right)
$$

<div align=center> <img width="550" src="../img/ch16/image 4.png"/> </div> <div align=center></div>

- **C 模型（Controller，控制器）**使用来自V模型和M模型的表示来选择合适的动作。控制器的目的是最大化期望的累积奖励。C是一个简单的单层线性模型，它将$z_t$和$h_t$直接映射到每个时间步的行动$a _t$，$ \left[z_t h_t\right] $是把 $z_t$和$h_t$拼接在一起

$$
a_t=W_c\left[z_t h_t\right]+b_c
$$

把 V、M、C 模型放一起，**整体运作的流程**是：在每个时间步$t$，原始观测输入到V，输出 $z_t $。输入到C的是隐向量$z_t $与M的隐状态$h_t $的拼接。接着，C会输出一个动作向量$a_t $用于运动控制，并且会影响环境。接着，M将当前的$z_t $和动作$a_t $作为输入，更新自身的隐状态，生成$h_{t+1} $。值得注意的是，论文中是通过随机策略跟环境进行交互收集到的预演（rollouts）来训练世界模型。

<div align=center> <img width="550" src="../img/ch16/image 5.png"/> </div> <div align=center></div>

按时间步展开来，世界模型的结构如下图所示。

<div align=center> <img width="550" src="../img/ch16/image 6.png"/> </div> <div align=center></div>



原论文给出了对应的世界模型 demo：https://worldmodels.github.io/，大家可以试玩下。也欢迎大家看我主页上世界模型相关的 paper（https://qiwang067.github.io/），最后宣传下我们 NeurIPS 组织的 Workshop “**Embodied World Models for Decision Making**”（https://embodied-world-models.github.io/）”，**Genie 3 的核心贡献者** Philip Ball 也会给 Talk，欢迎大家参加、投稿~，希望本文能对大家理解世界模型有所帮助。

<div align=center> <img width="550" src="../img/ch16/image 7.png"/> </div> <div align=center></div>



