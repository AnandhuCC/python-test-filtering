# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:34:47 2020

@author: ccana
"""
import numpy as np

def ax(m):
    t = m[1:3, 2:3]
    return t

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
a = np.matrix(a)
y = ax(a)
print(y)
