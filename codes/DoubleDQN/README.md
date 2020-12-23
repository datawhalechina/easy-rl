## 思路

见[博客](https://blog.csdn.net/JohnJim0/article/details/111552545)

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
可视化

```python
tensorboard --logdir logs 
```