# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:43:17 2022

@author: Erutalon
"""

#%%调用gpu
import tensorflow.compat.v1 as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 程序最多只能占用指定gpu70%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
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

"""
from tensorflow.keras import Input
#from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.callbacks import LearningRateScheduler

import sys
sys.path.append(r'F:\BS\Code\model\Train')
from MyModel import *
#%%
IMAGE_LIB = 'F:/BS/Kaggle_Measuring_Lungs/2d_images/'
MASK_LIB = 'F:/BS/Kaggle_Measuring_Lungs/2d_masks/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED=42

all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.tif']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im
    
#%%
fig, ax = plt.subplots(1,2, figsize = (8,4))
ax[0].imshow(x_data[0], cmap='gray')
ax[1].imshow(y_data[0], cmap='gray')
plt.show()
#%%
x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)

#%%
smooth=100
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
#%%

model = unet2()
#打印模型参数
model.summary()

model.load_weights('F:/BS/Code/model/Comparison/lung_256.h5')  #载入模型参数
model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])
#%%
def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
fix, ax = plt.subplots(8,2, figsize=(8,20))
for i in range(8):
    ax[i,0].imshow(image_batch[i,:,:,0])
    ax[i,1].imshow(mask_batch[i,:,:,0])
plt.show()

#%%


model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[iou,dice_coef])
"""
weight_saver = ModelCheckpoint('unet2_0429.h5', monitor='val_dice_coef',
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
#annealer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)


#%%训练模型
#8即为batchsize
hist = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 200,
                           validation_data = (x_val, y_val),
                           epochs=30, verbose=1,
                           callbacks = [weight_saver, annealer])
#%%
model.save('F:/BS/Code/model/220429/myUnet2_30ep.h5')
model.save_weights('F:/BS/Code/model/220428/myUnet2_30ep_weights_10.h5')
#%%
model.evaluate(x_val, y_val)


#%%精度和损失曲线

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['iou'], color='b')
plt.plot(hist.history['val_iou'], color='r')
plt.show()
plt.plot(hist.history['dice_coef'], color='b')
plt.plot(hist.history['val_dice_coef'], color='r')
plt.show()
"""
#%%预测
plt.imshow(model.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')
y_hat = model.predict(x_val)

#%%DenseCRF

from pydensecrf import densecrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral,create_pairwise_gaussian

IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED=42
"""
path="E:/Dataset/Collection/Set1/tr_im/tr_im/8.png"
im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
im = (im - np.min(im)) / (np.max(im) - np.min(im))
"""
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


#%%
i=8
im = x_val[i,:,:,:]
img_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
mask = y_val[i,:,:,:]
seg_map = y_hat[i,:,:,:]
final_mask = dense_crf(np.array(img_rgb).astype(np.uint8), seg_map)
fig, ax = plt.subplots(1,4,figsize=(12,12))
#ax[0].imshow(x_val[1,:,:,0], cmap='gray')
ax[0].imshow(x_val[i,:,:,0])
ax[1].imshow(y_val[i,:,:,0])
ax[2].imshow(y_hat[i,:,:,0])
ax[3].imshow(final_mask)
seg_dice=dice_coef(mask,seg_map)
final_mask=final_mask.astype('float32')
final_dice=dice_coef(mask,final_mask)
print("dice: \n seg_img = ",seg_dice, '\n',"densecrf = ",final_dice)
#%%UNet_DeneCRF

for i in range(134):
    im = x_val[i,:,:,:]
    img_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    mask = y_val[i,:,:,:]
    seg_map = y_hat[i,:,:,:]
    final_mask = dense_crf(np.array(img_rgb).astype(np.uint8), seg_map)
    fig, ax = plt.subplots(1,4,figsize=(12,12))
    #ax[0].imshow(x_val[1,:,:,0], cmap='gray')
    ax[0].imshow(x_val[i,:,:,0])
    ax[1].imshow(y_val[i,:,:,0])
    ax[2].imshow(y_hat[i,:,:,0])
    ax[3].imshow(final_mask)
    seg_dice=dice_coef(mask,seg_map)
    final_mask=final_mask.astype('float32')
    final_dice=dice_coef(mask,final_mask)
    print("dice: \n seg_img = ",seg_dice, '\n',"densecrf = ",final_dice)
    
"""
#%%

for i in range(134):
    im = x_val[i,:,:,:]
    img_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    final_mask = dense_crf(np.array(img_rgb).astype(np.uint8), seg_map)
    fig, ax = plt.subplots(1,4,figsize=(12,12))
    #ax[0].imshow(x_val[1,:,:,0], cmap='gray')
    ax[0].imshow(x_val[i,:,:,0])
    ax[1].imshow(y_val[i,:,:,0])
    ax[2].imshow(y_hat[i,:,:,0])
    ax[3].imshow(final_mask)

#model=model.load('F:/BS/Code/model/Train/dataset1.h5')
#预测
seg_map = model.predict(im.reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0]
img_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
final_mask = dense_crf(np.array(img_rgb).astype(np.uint8), seg_map)
#%%
plt.subplot(1, 3, 1)
plt.imshow(im)
plt.subplot(1, 3, 2)
plt.imshow(seg_map)
plt.subplot(1, 3, 3)
plt.imshow(final_mask)
plt.show()

"""
