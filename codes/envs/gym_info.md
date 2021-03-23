## 环境说明

### [CartPole v0](https://github.com/openai/gym/wiki/CartPole-v0)

<img src="assets/image-20200820174307301.png" alt="image-20200820174307301" style="zoom:50%;" />

通过向左或向右推车能够实现平衡，所以动作空间由两个动作组成。每进行一个step就会给一个reward，如果无法保持平衡那么done等于true，本次episode失败。理想状态下，每个episode至少能进行200个step，也就是说每个episode的reward总和至少为200，step数目至少为200

### [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0)

<img src="assets/image-20200820174814084.png" alt="image-20200820174814084" style="zoom:50%;" />

钟摆以随机位置开始，目标是将其摆动，使其保持向上直立。动作空间是连续的，值的区间为[-2,2]。每个step给的reward最低为-16.27，最高为0。目前最好的成绩是100个episode的reward之和为-123.11 ± 6.86。

### CliffWalking-v0

悬崖寻路问题（CliffWalking）是指在一个4 x 12的网格中，智能体以网格的左下角位置为起点，以网格的下角位置为终点，目标是移动智能体到达终点位置，智能体每次可以在上、下、左、右这4个方向中移动一步，每移动一步会得到-1单位的奖励。

<img src="./assets/image-20201007211441036.png" alt="image-20201007211441036" style="zoom:50%;" />

如图，红色部分表示悬崖，数字代表智能体能够观测到的位置信息，即observation，总共会有0-47等48个不同的值，智能体再移动中会有以下限制：

* 智能体不能移出网格，如果智能体想执行某个动作移出网格，那么这一步智能体不会移动，但是这个操作依然会得到-1单位的奖励

* 如果智能体“掉入悬崖” ，会立即回到起点位置，并得到-100单位的奖励

* 当智能体移动到终点时，该回合结束，该回合总奖励为各步奖励之和

实际的仿真界面如下：

<img src="./assets/image-20201007211858925.png" alt="image-20201007211858925" style="zoom:50%;" />

由于从起点到终点最少需要13步，每步得到-1的reward，因此最佳训练算法下，每个episode下reward总和应该为-13。