# Chapter10 Sparse Reward

## 1 Keywords

- **reward shaping：** 在我们的agent与environment进行交互时，我们人为的设计一些reward，从而“指挥”agent，告诉其采取哪一个action是最优的，而这个reward并不是environment对应的reward，这样可以提高我们estimate Q-function时的准确性。
- **ICM（intrinsic curiosity module）：** 其代表着curiosity driven这个技术中的增加新的reward function以后的reward function。
- **curriculum learning：** 一种广义的用在RL的训练agent的方法，其在input训练数据的时候，采取由易到难的顺序进行input，也就是认为设计它的学习过程，这个方法在ML和DL中都会普遍使用。
- **reverse curriculum learning：** 相较于上面的curriculum learning，其为更general的方法。其从最终最理想的state（我们称之为gold state）开始，依次去寻找距离gold state最近的state作为想让agent达到的阶段性的“理想”的state，当然我们应该在此过程中有意的去掉一些极端的case（太简单、太难的case）。综上，reverse curriculum learning 是从 gold state 去反推，就是说你原来的目标是长这个样子，我们从我们的目标去反推，所以这个叫做 reverse curriculum learning。  
- **hierarchical （分层） reinforcement learning：** 将一个大型的task，横向或者纵向的拆解成多个 agent去执行。其中，有一些agent 负责比较high level 的东西，负责订目标，然后它订完目标以后，再分配给其他的 agent把它执行完成。（看教程的 hierarchical  reinforcement learning部分的示例就会比较明了）

## 2 Questions

- 解决sparse reward的方法有哪些？

  答：Reward Shaping、curiosity driven reward、（reverse）curriculum learning 、Hierarchical Reinforcement learning等等。

- reward shaping方法存在什么主要问题？

  答：主要的一个问题是我们人为设计的reward需要domain knowledge，需要我们自己设计出符合environment与agent更好的交互的reward，这需要不少的经验知识，需要我们根据实际的效果进行调整。

- ICM是什么？我们应该如何设计这个ICM？

  答：ICM全称为intrinsic curiosity module。其代表着curiosity driven这个技术中的增加新的reward function以后的reward function。具体来说，ICM在更新计算时会考虑三个新的东西，分别是 state $s_1$、action $a_1$ 和 state $s_2$。根据$s_1$ 、$a_1$、 $a_2$，它会 output 另外一个新的 reward $r_1^i$。所以在ICM中我们total reward 并不是只有 r 而已，还有 $r^i$。它不是只有把所有的 r 都加起来，它还把所有 $r^i$ 加起来当作total reward。所以，它在跟环境互动的时候，它不是只希望 r 越大越好，它还同时希望 $r^i$ 越大越好，它希望从 ICM 的 module 里面得到的 reward 越大越好。ICM 就代表了一种curiosity。

  对于如何设计ICM，ICM的input就像前面所说的一样包括三部分input 现在的 state $s_1$，input 在这个 state 采取的 action $a_1$，然后接 input 下一个 state $s_{t+1}$，对应的output就是reward $r_1^i$，input到output的映射是通过network构建的，其使用 $s_1$ 和 $a_1$ 去预测 $\hat{s}_{t+1}$ ,然后继续评判预测的$\hat{s}_{t+1}$和真实的$s_{t+1}$像不像，越不相同得到的reward就越大。通俗来说这个reward就是，如果未来的状态越难被预测的话，那么得到的reward就越大。这也就是curiosity的机制，倾向于让agent做一些风险比较大的action，从而增加其machine exploration的能力。

  同时为了进一步增强network的表达能力，我们通常讲ICM的input优化为feature extractor，这个feature extractor模型的input就是state，output是一个特征向量，其可以表示这个state最主要、重要的特征，把没有意义的东西过滤掉。
