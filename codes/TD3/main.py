#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 23:38:13
@LastEditor: John
@LastEditTime: 2020-06-11 23:38:31
@Discription: 
@Environment: python 3.7.7
'''
import torch
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")