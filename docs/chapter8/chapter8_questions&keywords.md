# Chapter8 Q-learning for Continuous Actions

## Questions

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
