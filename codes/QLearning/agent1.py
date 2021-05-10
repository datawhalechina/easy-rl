#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-04-29 17:02:00
Discription: 
Environment: 
'''
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import math
#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2021-04-29 16:45:33
Discription: use np array to define Q table
Environment: 
'''
import numpy as np
import math

class QLearning(object):
    def __init__(self,
                 state_dim,action_dim,cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  
        self.epsilon = 0 
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = np.zeros((state_dim, action_dim)) # Q表
        
    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:  # 随机选取0-1之间的值，如果大于epsilon就按照贪心策略选取action，否则随机选取
            action = self.predict(state)
        else:
            action = np.random.choice(self.action_dim)  #有一定概率随机探索选取一个动作
        return action

    def predict(self, state):
        '''根据输入观测值，采样输出的动作值，带探索，测试模型时使用
        '''
        Q_list = self.Q_table[state, :]
        Q_max = np.max(Q_list)
        action_list = np.where(Q_list == Q_max)[0]  
        action = np.random.choice(action_list) # Q_max可能对应多个 action ，可以随机抽取一个
        return action
            
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[state, action]
        if done:
            Q_target = reward  # 没有下一个状态了
        else:
            Q_target = reward + self.gamma * np.max(
                self.Q_table[next_state, :])  # Q_table-learning
        self.Q_table[state, action] += self.lr * (Q_target - Q_predict)  # 修正q
    def save(self,path):
        np.save(path+"Q_table.npy", self.Q_table)
    def load(self, path):
        self.Q_table = np.load(path+"Q_table.npy")


