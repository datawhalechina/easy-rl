#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-03-25 23:25:15
LastEditor: JiangJi
LastEditTime: 2021-04-28 21:36:50
Discription: 
Environment: 
'''
import random
dic = {0:"鳗鱼家",1:"一心",2:"bada"}
print("0:鳗鱼家，1:一心，2:bada")
print("三次随机，取最后一次选择")
for i in range(3):
    if i ==2:
        print(f"去{dic[random.randint(0,2)]}")
    else:
        print(f"不去{dic[random.randint(0,2)]}")