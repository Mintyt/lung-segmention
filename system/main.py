# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:39:57 2022

@author: Erutalon
"""
#调用gpu
import tensorflow.compat.v1 as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 程序最多只能占用指定gpu70%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)


from easygui import *
import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sys.path.append(r'F:\BS\system')
import predict

while(1):
    path = fileopenbox()
    predict.segmention(path)
    if buttonbox('继续？', choices=('YES', 'NO'))=='NO':
        sys.exit(0)
