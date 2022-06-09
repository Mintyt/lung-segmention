# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:01:29 2022

@author: Erutalon
"""
#%%
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

from pydensecrf import densecrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral,create_pairwise_gaussian
#%%
"""
#%%测试
segmention('F:\BS\Kaggle_Measuring_Lungs\2d_images\ID_0002_Z_0162.tif')
img_path='F:\BS\Kaggle_Measuring_Lungs\2d_images\ID_0002_Z_0162.tif'
img_pathimg_path=img_path.replace('\\','/')
#%%
"""
def segmention(img_path):
    #导入图片
    IMG_HEIGHT, IMG_WIDTH = 256, 256
    SEED=42
    #img_path=img_path.replace('\\','/')
    img = imgrshape(img_path)
    #加载网络
    model = unet2()
    model.load_weights('F:/BS/Code/model/Comparison/lung_256.h5')  #载入模型参数
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
    
    #预测
    seg_map = model.predict(img.reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0]
    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    final_mask = dense_crf(np.array(img_rgb).astype(np.uint8), seg_map)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(seg_map)
    plt.subplot(1, 3, 3)
    plt.imshow(final_mask)
    plt.show()
    
    
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def unet2(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(8, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(8, (3, 3), padding='same')(bn1)
    bn1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(16, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(16, (3, 3), padding='same')(bn2)
    bn2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(32, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(32, (3, 3), padding='same')(bn3)
    bn3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(32, (1, 1), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    
    up1 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bn4), conv3])
    conv5 = Conv2D(32, (2, 2), padding='same')(up1)
    bn5 = Activation('relu')(conv5)
    
    up2 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(bn5), conv2])
    conv6 = Conv2D(32, (2, 2), padding='same')(up2)
    bn6 = Activation('relu')(conv6)
    
    up3 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(bn6), conv1])
    conv7 = Conv2D(16, (2, 2), padding='same')(up3)
    bn7 = Activation('relu')(conv7)
    
    conv8 = Conv2D(64, (1, 1), padding='same')(bn7)
    bn8 = Activation('relu')(conv8)
    
    D = Dropout(0.5)(bn8)

    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(D)

    return Model(inputs=[inputs], outputs=[output_layer])

def imgrshape(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(256,256), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im

#DenseCRF
def dense_crf(img, output_probs): # img 为H，*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），#seg_map - 假设为语义分割的 mask, hxw, np.array 形式.

    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(1)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

#%%可视化mask
"""
def maskplot(img1,prediction):
    #index=np.random.randint(1,len(df_test.index))
    #img1 = cv2.imread(df_test['filename'].iloc[index])
    img1 = cv2.resize(img1 ,(256, 256))
    img = img1 / 255
    img = img[np.newaxis, :, :, :]
    #prediction=model.predict(img)
    prediction = np.squeeze(prediction)
    ground_truth = cv2.resize(cv2.imread(df_test['mask'].iloc[index]),(256,256)).astype("uint8")
    ground_truth = cv2.cvtColor(ground_truth,cv2.COLOR_BGR2GRAY) 
    _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
    contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img1, contours_gt, -1, (0, 255, 0),2)

    prediction[np.nonzero(prediction < 0.7)] = 0.0
    prediction[np.nonzero(prediction >=0.7)] = 255.
    prediction = prediction.astype("uint8")
    _, thresh_p = cv2.threshold(prediction, 127, 255, 0)
    contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img1, contours_p, -1, (255,36,0),2)
    
    return overlap_img
"""