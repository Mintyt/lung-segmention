# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:00:15 2022

@author: Erutalon
"""
#%%
from easygui import *
import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#%%
sys.path.append(r'F:\BS\system')
import predict

enter = msgbox(msg='欢迎进入图像辅助系统',title='医学图像辅助查看系统',ok_button='进入')
path = ''
while(enter == '进入'or'继续'):
    enterpath=buttonbox(msg='欢迎进入图像辅助系统',title='医学图像辅助查看系统',choices=('手动输入', '浏览'))
    if enterpath=='手动输入':
        path = enterbox(msg='选择图片',title='医学图像辅助查看系统')
    elif enterpath=='浏览':
        path = fileopenbox()
    else :
        msgbox(msg = 'Error',title='医学图像辅助查看系统')
    
    if path!='':
        output = predict.segmention(path)
        enter=buttonbox(title='医学图像辅助查看系统', image='filename.png', choices=('继续', '推出'))
    
        
"""
#%%           
import easygui
easygui.egdemo()                
"""                
                