#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-25 23:25:15
LastEditor: John
LastEditTime: 2021-03-26 16:46:52
Discription: 
Environment: 
'''
from collections import defaultdict
import numpy as np
action_dim = 2
Q_table  = defaultdict(lambda: np.zeros(action_dim))
Q_table[str(0)] = 1
print(Q_table[str(0)])
Q_table[str(21)] = 3
print(Q_table[str(21)])