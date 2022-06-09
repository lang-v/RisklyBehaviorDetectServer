import io
import os
import time

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Activation, MaxPool3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, ZeroPadding3D

# 两种方法构建的是大致相同的模型
# 并没有验证C3Dnet是否可用，应该是没有问题的。


def C3Dnet(nb_classes, input_shape):
    weight_decay = 0.005
    inputs = keras.Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model


def C3D_model(classes_num, input_shape):
    c3d_model = keras.Sequential([
        # input shape (FRAME_SIZE, IMAGE_W, IMAGE_H, 3),
        tf.keras.layers.Rescaling(1. / 255, input_shape=input_shape),
        keras.layers.Conv3D(64, 3, activation='relu',input_shape=input_shape),
        keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'),
        keras.layers.Conv3D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
        keras.layers.Conv3D(256, 3, padding='same', activation='relu'),
        keras.layers.Conv3D(256, 3, padding='same', activation='relu'),
        keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
        keras.layers.Conv3D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv3D(512, 3, padding='same', activation='relu'),
        keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
        keras.layers.Conv3D(512, 3, padding='same', activation='relu'),
        keras.layers.Conv3D(512, 3, padding='same', activation='relu'),
        keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(4096,activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096,activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(classes_num, activation='softmax')
    ])

    c3d_model.summary()
    print(c3d_model.output)
    return c3d_model
