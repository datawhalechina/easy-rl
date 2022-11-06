#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-10-26 07:38:17
Discription: 
Environment: 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
from functools import wraps
from time import time
import logging
from pathlib import Path


from matplotlib.font_manager import FontProperties  # 导入字体模块

def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
        fname='/System/Library/Fonts/STHeiti Light.ttc', size=15) # fname系统字体路径，此处是mac的
    except:
        font = None
    return font

def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg.env_name,
              cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(cfg.result_path+f"{tag}_rewards_curve_cn")
    # plt.show()
def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,title="learning curve",fpath=None,save_fig=True,show_fig=False):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    if save_fig:
        plt.savefig(f"{fpath}/learning_curve.png")
    if show_fig:
        plt.show()

def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()

def save_results(res_dic,fpath = None):
    ''' save results
    '''
    Path(fpath).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{fpath}/res.csv",index=None)
def merge_class_attrs(ob1, ob2):
    ob1.__dict__.update(ob2.__dict__)
    return ob1
def get_logger(fpath):
    Path(fpath).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name='r')  # set root logger if not set name
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # output to file by using FileHandler
    fh = logging.FileHandler(fpath+"log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # output to screen by using StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
def save_cfgs(cfgs, fpath):
    ''' save config
    '''
    Path(fpath).mkdir(parents=True, exist_ok=True)
 
    with open(f"{fpath}/config.yaml", 'w') as f:
        for cfg_type in cfgs:
            yaml.dump({cfg_type: cfgs[cfg_type].__dict__}, f, default_flow_style=False)
def load_cfgs(cfgs, fpath):
    with open(fpath) as f:
        load_cfg = yaml.load(f,Loader=yaml.FullLoader)
        for cfg_type in cfgs:
            for k, v in load_cfg[cfg_type].items():
                setattr(cfgs[cfg_type], k, v)
# def del_empty_dir(*paths):
#     ''' 删除目录下所有空文件夹
#     '''
#     for path in paths:
#         dirs = os.listdir(path)
#         for dir in dirs:
#             if not os.listdir(os.path.join(path, dir)):
#                 os.removedirs(os.path.join(path, dir))

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
        
# def save_args(args,path=None):
#     # save parameters  
#     Path(path).mkdir(parents=True, exist_ok=True) 
#     with open(f"{path}/params.json", 'w') as fp:
#         json.dump(args, fp,cls=NpEncoder)   
#     print("Parameters saved!")


def timing(func):
    ''' a decorator to print the running time of a function
    ''' 
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print(f"func: {func.__name__}, took: {te-ts:2.4f} seconds")
        return result
    return wrap
def all_seed(env,seed = 1):
    ''' omnipotent seed for RL, attention the position of seed function, you'd better put it just following the env create function
    Args:
        env (_type_): 
        seed (int, optional): _description_. Defaults to 1.
    '''
    import torch
    import numpy as np
    import random
    # print(f"seed = {seed}")
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    