## 0. 写在前面

本项目用于学习RL基础算法，主要面向对象为RL初学者、需要结合RL的非专业学习者，尽量做到: **注释详细**，**结构清晰**。

注意本项目为实战内容，建议首先掌握相关算法的一些理论基础，再来享用本项目，理论教程参考本人参与编写的[蘑菇书](https://github.com/datawhalechina/easy-rl)。

未来开发计划包括但不限于：多智能体算法、强化学习Python包以及强化学习图形化编程平台等等。

## 1. 项目说明

项目内容主要包含以下几个部分：
* [Jupyter Notebook](./notebooks/)：使用Notebook写的算法，有比较详细的实战引导，推荐新手食用
* [codes](./codes/)：这些是基于Python脚本写的算法，风格比较接近实际项目的写法，推荐有一定代码基础的人阅读，下面会说明其具体的一些架构
* [附件](./assets/)：目前包含强化学习各算法的中文伪代码


[codes](./assets/)结构主要分为以下几个脚本：
* ```[algorithm_name].py```：即保存算法的脚本，例如```dqn.py```，每种算法都会有一定的基础模块，例如```Replay Buffer```、```MLP```(多层感知机)等等；
* ```task.py```: 即保存任务的脚本，基本包括基于```argparse```模块的参数，训练以及测试函数等等，其中训练函数即```train```遵循伪代码而设计，想读懂代码可从该函数入手；
* ```utils.py```：该脚本用于保存诸如存储结果以及画图的软件，在实际项目或研究中，推荐大家使用```Tensorboard```来保存结果，然后使用诸如```matplotlib```以及```seabron```来进一步画图。
## 2. 算法列表

注：点击对应的名称会跳到[codes](./codes/)下对应的算法中，其他版本还请读者自行翻阅

|                算法名称                 |                           参考文献                           | 备注 |
| :-------------------------------------: | :----------------------------------------------------------: | :--: |
| [Policy Gradient](codes/PolicyGradient) | [Policy Gradient paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) |      |
|                 DQN-CNN                 |                                                              | 待更 |
|      [DoubleDQN](codes/DoubleDQN)       |     [Double DQN Paper](https://arxiv.org/abs/1509.06461)     |      |
|          [SoftQ](codes/SoftQ)           |  [Soft Q-learning paper](https://arxiv.org/abs/1702.08165)   |      |
|            [SAC](codes/SAC)             |      [SAC paper](https://arxiv.org/pdf/1812.05905.pdf)       |      |
|        [SAC-Discrete](codes/SAC)        |  [SAC-Discrete paper](https://arxiv.org/pdf/1910.07207.pdf)  |      |
|                  SAC-S                  |       [SAC-S paper](https://arxiv.org/abs/1801.01290)        |      |
|                  DSAC                   | [DSAC paper](https://paperswithcode.com/paper/addressing-value-estimation-errors-in) | 待更 |

## 3. 算法环境

算法环境说明请跳转[env](./codes/envs/README.md)

## 4. 运行环境

主要依赖：Python 3.7、PyTorch 1.10.0、Gym 0.25.2。

### 4.1. 创建Conda环境
```bash
conda create -n easyrl python=3.7
conda activate easyrl # 激活环境
```
### 4.2. 安装Torch

安装CPU版本：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
```
安装CUDA版本：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
如果安装Torch需要镜像加速的话，点击[清华镜像链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/)，选择对应的操作系统，如```win-64```，然后复制链接，执行：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
```
也可以使用PiP镜像安装（仅限CUDA版本）：
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
### 4.3. 检验CUDA版本Torch安装

CPU版本Torch请忽略此步，执行如下Python脚本，如果返回True说明CUDA版本安装成功:
```python
import torch
print(torch.cuda.is_available())
```
### 4.4. 安装Gym

```bash
pip install gym==0.25.2
```
如需安装Atari环境，则需另外安装

```bash
pip install gym[atari,accept-rom-license]==0.25.2
```

### 4.5. 安装其他依赖

项目根目录下执行：
```bash
pip install -r requirements.txt
```

## 6.使用说明

对于[codes](./codes/)，`cd`到对应的算法目录下，例如`DQN`：

```bash
python task_0.py
```

或者加载配置文件：

```bash
python task0.py --yaml configs/CartPole-v1_DQN_Train.yaml
```

对于[Jupyter Notebook](./notebooks/)：

* 直接运行对应的ipynb文件就行

## 6. 友情说明

推荐使用VS Code做项目，入门可参考[VSCode上手指南](https://blog.csdn.net/JohnJim0/article/details/126366454)