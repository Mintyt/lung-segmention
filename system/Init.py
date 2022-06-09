# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:12:56 2022

@author: Erutalon
"""

def init():
    import os
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import cv2
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.layers import *
    from keras.regularizers import l2
    from skimage.color import rgb2gray
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model, load_model, save_model
    from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    from tensorflow.keras import backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow import keras
    from keras.callbacks import LearningRateScheduler