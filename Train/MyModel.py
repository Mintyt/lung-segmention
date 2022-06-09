# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:35:36 2022

@author: Erutalon
"""
from keras.layers import *
#from tensorflow.keras import Input
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout,  Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, add, UpSampling2D

#对比模型完整版
def unet2pro(input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
#对比论文模型
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
#脑肿瘤模型
def unet(input_size=(256,256,3)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])
#FCN8s
def FCN8s(nClasses,input_size=(256,256,1)):
    
    inputs = Input(input_size)
###编码器部分
    conv1 = Conv2D(filters=32, input_shape=input_size,
                   kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv1')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv1')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv1')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv2')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    score_pool3 = Conv2D(filters=nClasses, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool3')(pool3)#此行代码为后面的跳层连接做准备

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv1')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv2')(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    score_pool4 = Conv2D(filters=nClasses, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool4')(pool4)#此行代码为后面的跳层连接做准备

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv1')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv2')(conv5)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)
###1×1卷积部分，加入了Dropout层以免过拟合
    fc6 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc6')(pool5)
    fc6 = Dropout(0.3, name='dropout_1')(fc6)

    fc7 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc7')(fc6)
    fc7 = Dropout(0.3, name='dropour_2')(fc7)
###下面的代码为跳层连接结构
    score_fr = Conv2D(filters=nClasses, kernel_size=(1, 1), padding='same',
                      activation='relu', name='score_fr')(fc7)

    score2 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score2")(score_fr)

    add1 = add(inputs=[score2, score_pool4], name="add_1")

    score4 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score4")(add1)

    add2 = add(inputs=[score4, score_pool3], name="add_2")

    UpSample = Conv2DTranspose(filters=nClasses, kernel_size=(8, 8), strides=(8, 8),
                               padding="valid", activation=None,
                               name="UpSample")(add2)

    outputs = Conv2D(1, 1, activation='sigmoid')(UpSample)

    return Model(inputs=[inputs], outputs=[outputs])
#FCN32s
def FCN32s(input_size=(256,256,1)):
    
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3),padding = "same")(inputs)
    bn1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(bn1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    bn2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    bn3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    bn4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    bn5 = Activation('relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn5)

    up1 = UpSampling2D(size=(32, 32), interpolation="bilinear")(pool5)

    conv6 = Conv2D(1, (1, 1), padding="same")(up1)
    output_layer = Activation('sigmoid')(conv6)

    return Model(inputs=[inputs], outputs=[output_layer])
#FCN16s
def FCN16s(input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(filters=32, input_shape=input_size,
                   kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv1')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv1')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv1')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv1')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)
    # max_pool_5 转置卷积上采样 2 倍至 max_pool_4 一样大

    #score_fr = Conv2D(filters=nClasses, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr')(pool5)
  
    up1 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="up1")(pool5)

    add1 = add(inputs=[pool4, up1])

    # _16s 上采样 16 倍后与输入尺寸相同
    up2 = UpSampling2D(size=(16, 16), interpolation="bilinear")(add1)

    # 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
    output_layer = Conv2D(1, kernel_size=(1, 1), activation="sigmoid", padding="same")(up2)

    return Model(inputs=[inputs], outputs=[output_layer])

#help
def help():
    print('论文对比模型：unet2  input_size=(256,256,1)\n',
          '脑肿瘤模型：unet  input_size=(256,256,3)\n',
          'FCN8s(nClasses,input_size=(256,256,1),其中nClasses为分类数目\n',
          'FCN32s(input_size=(256,256,1))\n',
          'FCN16s(input_size=(256,256,1))\n'
          )
    
    
    
    
    