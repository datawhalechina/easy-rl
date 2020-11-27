# Policy Gradient
实现的是Policy Gradient最基本的REINFORCE方法
## 原理讲解

参考我的博客[Policy Gradient算法实战](https://blog.csdn.net/JohnJim0/article/details/110236851)

## 环境

python 3.7.9

pytorch 1.6.0

tensorboard 2.3.0 

torchvision 0.7.0 

## 程序运行方法

train: 

```python
python main.py 
```

eval: 

```python
python main.py --train 0 
```
tensorboard：
```python
tensorboard --logdir logs 
```


## 参考

[REINFORCE和Reparameterization Trick](https://blog.csdn.net/JohnJim0/article/details/110230703)

[Policy Gradient paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

[REINFORCE](https://towardsdatascience.com/policy-gradient-methods-104c783251e0)