## 思路

见[我的博客](https://blog.csdn.net/JohnJim0/article/details/109557173)
## 环境

python 3.7.9

pytorch 1.6.0

tensorboard 2.3.0 

torchvision 0.7.0 

## 使用

train: 

```python
python main.py 
```

eval: 

```python
python main.py --train 0 
```
可视化：
```python
tensorboard --logdir logs 
```

## Torch知识

[with torch.no_grad()](https://www.jianshu.com/p/1cea017f5d11)

