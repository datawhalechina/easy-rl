## 环境说明

### [CartPole v0](https://github.com/openai/gym/wiki/CartPole-v0)

<img src="assets/image-20200820174307301.png" alt="image-20200820174307301" style="zoom:50%;" />

通过向左或向右推车能够实现平衡，所以动作空间由两个动作组成。每进行一个step就会给一个reward，如果无法保持平衡那么done等于true，本次episode失败。理想状态下，每个episode至少能进行200个step，也就是说每个episode的reward总和至少为200，step数目至少为200

### [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0)

<img src="assets/image-20200820174814084.png" alt="image-20200820174814084" style="zoom:50%;" />

钟摆以随机位置开始，目标是将其摆动，使其保持向上直立。动作空间是连续的，值的区间为[-2,2]。每个step给的reward最低为-16.27，最高为0。目前最好的成绩是100个episode的reward之和为-123.11 ± 6.86。