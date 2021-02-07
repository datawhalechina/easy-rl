# Chapter11 Imitation Learning 

## 1 Keywords

- **Imitation learning：**  其讨论我们没有reward或者无法定义reward但是有与environment进行交互时怎么进行agent的学习。这与我们平时处理的问题中的情况有些类似，因为通常我们无法从环境中得到明确的reward。Imitation learning 又被称为 learning from demonstration (示范学习) ，apprenticeship learning (学徒学习)，learning by watching (观察学习)等。
- **Behavior Cloning：** 类似于ML中的监督学习，通过收集expert的state与action的对应信息，训练我们的network（actor）。在使用时input state时，得到对应的outpur action。
- **Dataset Aggregation：** 用来应对在Behavior Cloning中expert提供不到的data，其希望收集expert在各种极端state下expert的action。
- **Inverse Reinforcement learning（IRL）：** Inverse Reinforcement Learning 是先找出 reward function，再去用 Reinforcement Learning 找出 optimal actor。这么做是因为我们没有环境中reward，但是我们有expert 的demonstration，使用IRL，我们可以推断expert 是因为什么样的 reward function 才会采取这些action。有了reward function 以后，接下来，就可以套用一般的 reinforcement learning 的方法去找出 optimal actor。
- **Third Person Imitation Learning：** 一种把第三人称视角所观察到的经验 generalize 到第一人称视角的经验的技术。

## 2 Questions

- 对于Imitation Learning 的方法有哪些？

  答：Behavior Cloning、Inverse Reinforcement Learning（IRL）或者称为Inverse Optimal Control。

- Behavior Cloning存在哪些问题呢？我们可以如何处理呢？

  答：

  1. 首先，如果只收集expert 的data（看到某一个state输出的action），你可能看过的 observation 会是非常 limited。所以我们要收集expert在各种极端state下的action，或者说是要收集更多的、复杂的data，可以使用教程中提到的Dataset Aggregation。
  2. 另外，使用传统意义上的Behavior Cloning的话，机器会完全 copy expert 的行为，不管 expert 的行为是否有道理，就算没有道理，没有什么用的，这是expert 本身的习惯，机器也会硬把它记下来。我们的agent是一个 machine，它是一个 network，network 的capacity 是有限的。就算给 network training data，它在training data 上得到的正确率往往也不是100%，他有些事情是学不起来的。这个时候，什么该学，什么不该学就变得很重要。不过极少数expert的行为是没有意义的，但是至少也不会产生较坏的影响。
  3. 还有，在做 Behavior Cloning 的时候，training data 跟 testing data 是 mismatch 的。我们可以用 Dataset Aggregation 的方法来缓解这个问题。这个问题是，在 training 跟 testing 的时候，data distribution 其实是不一样的。因为在 reinforcement learning 里面，action 会影响到接下来所看到的 state。我们是先有 state $s_1$，然后采取 action $a_1$，action $a_1$ 其实会决定接下来你看到什么样的 state $s_2$。所以在 reinforcement learning 里面有一个很重要的特征，就是你采取了 action 会影响你接下来所看到的 state。如果做了Behavior Cloning 的话，我们只能观察到 expert 的一堆 state 跟 action 的pair。然后我们希望可以 learn 一个 $\pi^*$，我们希望 $\pi^*$ 跟 $\hat{\pi}$ 越接近越好。如果 $\pi^*$ 可以跟 $\hat{\pi}$ 一模一样的话，你 training 的时候看到的 state 跟 testing 的时候所看到的 state 会是一样的，这样模型的泛化性能就会变得比较差。而且，如果你的 $\pi^*$ 跟 $\hat{\pi}$ 有一点误差。这个误差在一般 supervised learning problem 里面，每一个 example 都是 independent 的，也许还好。但对 reinforcement learning 的 problem 来说，可能在某个地方，也许 machine 没有办法完全复制 expert 的行为，也许最后得到的结果就会差很多。所以 Behavior Cloning 并不能够完全解决 Imatation learning 这件事情，我们可以使用另外一个比较好的做法叫做 Inverse Reinforcement Learning。


- Inverse Reinforcement Learning 是怎么运行的呢？

  答：首先，我们有一个 expert $\hat{\pi}$，这个 expert 去跟环境互动，给我们很多 $\hat{\tau_1}$ 到 $\hat{\tau_n}$，我们需要将其中的state、action这个序列都记录下来。然后对于actor $\pi$ 也需要进行一样的互动和序列的记录。接着我们需要指定一个reward function，并且保证expert对应的分数一定要比actor的要高，用过这个reward function继续learning更新我们的训练并且套用一般条件下的RL方法进行actor的更新。在这个过程中，我们也要同时进行我们一开始制定的reward function的更新，使得actor得得分越来越高，但是不超过expert的得分。最终的reward function应该让expert和actor对应的reward function都达到比较高的分数，并且从最终的reward function中无法分辨出谁应该得到比较高的分数。

- Inverse Reinforcement Learning 方法与GAN在图像生成中有什么异曲同工之处?

  答：在GAN 中，我们有一些比较好的图片数据集，也有一个generator，一开始他根本不知道要产生什么样的图，只能随机生成。另外我们有一个discriminator，其用来给生成的图打分，expert 生成的图得分高，generator 生成的图得分低。有了discriminator 以后，generator 会想办法去骗过 discriminator。Generator 会希望 discriminator 也会给它生成得图高分。整个 process 跟 IRL 的过程是类似的。我们一一对应起来看：

  * 生成的图就是 expert 的 demonstration，generator 就是actor，generator 会生成很多的图并让actor 与环境进行互动，从而产生很多 trajectory。这些 trajectory 跟环境互动的记录等价于 GAN 里面的生成图。
  * 在IRL中 learn 的 reward function 就是 discriminator。Rewards function 要给 expert 的 demonstration 高分，给 actor 互动的结果低分。
  * 考虑两者的过程，在IRL中，actor 会想办法，从这个已经 learn 出来的 reward function 里面得到高分，然后 iterative 地去循环这其实是与 GAN 的过程是一致的。
