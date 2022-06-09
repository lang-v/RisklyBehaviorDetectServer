import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, Input
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, ZeroPadding3D, Reshape, MaxPool3D

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


# 定义多输入多输出模型,
# 将两个模型绑在一块,但是训练也需要一起训练；还是决定采用分离训练
def mutil_output_c3d_model(action_classes_num, action_input_shape, scene_classes_num, scene_input_shape, lr=0.01):
    # 行为识别模型
    action_input_tensor = Input(shape=action_input_shape)
    # 1st block
    # x = Rescaling(1. / 255)(input_tensor)
    x = Conv3D(64, 3, activation='relu', name='conv1')(action_input_tensor)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='pool1')(x)
    # 2nd block
    x = Conv3D(128, 3, activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool2')(x)
    # 3rd block
    x = Conv3D(256, 3, activation='relu', padding='same', name='conv3a')(x)
    x = Conv3D(256, 3, activation='relu', padding='same', name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool3')(x)
    # 4th block
    x = Conv3D(512, 3, activation='relu', padding='same', name='conv4a')(x)
    x = Conv3D(512, 3, activation='relu', padding='same', name='conv4b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool4')(x)
    # 5th block
    x = Conv3D(512, 3, activation='relu', padding='same', name='conv5a')(x)
    x = Conv3D(512, 3, activation='relu', padding='same', name='conv5b')(x)
    # x = ZeroPadding3D(padding=(0, 1, 1), name='zeropadding')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool5')(x)
    # full connection
    # x = Reshape([512 * 4 * 4])(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    fc_action = keras.layers.Dense(name='fc_action', units=action_classes_num, activation='softmax')(x)

    # 场景识别模型
    scene_input_tensor = Input(shape=scene_input_shape)
    x2 = Conv2D(16, 3, padding='same', activation='relu')(scene_input_tensor)
    x2 = MaxPooling2D()(x2)
    x2 = Conv2D(32, 3, padding='same', activation='relu')(x2)
    x2 = MaxPooling2D()(x2)
    x2 = Conv2D(64, 3, padding='same', activation='relu')(x2)
    x2 = MaxPooling2D()(x2)
    x2 = Flatten()(x2)
    x2 = Dense(128, activation='relu')(x2)
    fc_scene = keras.layers.Dense(name='fc_scene', units=scene_classes_num, activation='softmax')(x2)

    model = Model(inputs=[action_input_tensor, scene_input_tensor], outputs=[fc_action, fc_scene])

    losses = {'fc_action': 'categorical_crossentropy','fc_scene': 'categorical_crossentropy'}
    loss_weights = {'fc_action': 1.0,'fc_scene': 1.0}
    adam = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=adam, loss=losses, loss_weights=loss_weights, metrics=['acc'])

    return model
