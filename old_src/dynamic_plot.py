#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import matplotlib.pyplot as plt
import numpy as np

#import warnings
## 解决(MatplotlibDeprecationWarning: Using default event loop until function specific to this GUI is implemented warnings.warn(str, mplDeprecation)
#warnings.filterwarnings("ignore",".*GUI is implemented.*")

a = 2
b = 3
def main():
    fig = plt.figure()
    # 连续性的画图
    ax = fig.add_subplot(1, 1, 1)
    # 设置图像显示的时候XY轴比例
    #ax.axis("equal")
    # 开启一个画图的窗口
    
    x = -1
    for i in range(500):
        #plt.ion()
        x += 1
        y = np.random.randn()
        #ax.scatter(x, y, c='r')#, marker='v')
        ax.bar(x, y, label='mean reward', color="r")
        plt.pause(0.0000000000000000000000001)
        c = a + b
        #plt.show()
        
#except Exception as err:
#    print(err)
# 防止运行结束时闪退
#plt.pause(0)