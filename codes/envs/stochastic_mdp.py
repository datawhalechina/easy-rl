#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-24 22:12:19
LastEditor: John
LastEditTime: 2021-03-26 17:12:43
Discription: 
Environment: 
'''
import numpy as np
import random


class StochasticMDP:
    def __init__(self):
        self.end = False
        self.curr_state = 2
        self.action_dim = 2
        self.state_dim = 6
        self.p_right = 0.5

    def reset(self):
        self.end = False
        self.curr_state = 2
        state = np.zeros(self.state_dim)
        state[self.curr_state - 1] = 1.
        return state

    def step(self, action):
        if self.curr_state != 1:
            if action == 1:
                if random.random() < self.p_right and self.curr_state < self.state_dim:
                    self.curr_state += 1
                else:
                    self.curr_state -= 1

            if action == 0:
                self.curr_state -= 1
        if self.curr_state == self.state_dim:
            self.end = True

        state = np.zeros(self.state_dim)
        state[self.curr_state - 1] = 1.

        if self.curr_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00/100.00, True, {}
        else:
            return state, 0.0, False, {}
