## Asynchronous Methods for Deep Reinforcement Learning

[论文地址](https://arxiv.org/abs/1602.01783)

本文贡献在于提出一种异步梯度优化框架，在各类型rl算法上都有较好效果，其中效果最好的就是（asynchronous advantage actor-critic, A3C）。

- **Motivation(Why)**:基于经验回放的drl算法在一些任务上效果较差，可以采用其他的方式来使用数据训练网络
- **Main Idea(What)**:使用异步并行的方式让多个agent与环境进行交互，当某一个agent交互完计算梯度后就交给global进行更新，更新完再同步给这个agent，其他agent做同样的操作。

这里给张李宏毅老师的讲解图：

![image-20221109213901773](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20221109213901773.png)



框架思想较为简单，但优点很多：

1. 适用范围广，on/off policy、离散/连续型任务均可使用

2. 多个agent采样数据减少数据相关性，训练更稳定

3. 不需要经验回放，让框架不局限于off-policy，同时减少内存算力消耗

4. 相较分布式所需资源更少，不用gpu，使用一台多核cpu机器时间更短

   

   本文使用hogwild! 方法来更新梯度，不清楚的可参考[知乎讲解](https://zhuanlan.zhihu.com/p/30826161)，下图来源此文：

   

   ![image-20221109221145358](https://gitee.com/xyfcw/CloudPictures/raw/master/img/image-20221109221145358.png)

   

   

这里举个栗子：假设网络有100个参数，worker i传来的梯度更新了40个参数后worker j就开始更新了，当$i$更新完后前面的一些参数被j更新了（被覆盖掉了），但不要紧，i更新完了照样把当前global参数同步给i，这就是Asynchronous的意思。

A3C：

![image-20221109234156910](https://gitee.com/xyfcw/CloudPictures/raw/master/img/202211092341970.png)

实验结果:

![image-20221110000219965](https://gitee.com/xyfcw/CloudPictures/raw/master/img/202211100002021.png)