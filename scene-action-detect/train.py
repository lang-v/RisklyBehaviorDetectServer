import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import os
from processing import get_all_dataset, get_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import model as c3d

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# scene data
SCENE_CLASSES_NUM = 3
SCENE_IMAGE_W = 360
SCENE_IMAGE_H = 360

# action data
ACTION_CLASSES_NUM = 4
ACTION_IMAGE_W = 112
ACTION_IMAGE_H = 112


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False, aug=None):
    while 1:  # 要无限循环
        assert len(inputs) == len(targets[0]) == len(targets[1])  # 判断输入数据长度和label长度是否相同
        for start_idx in range(len(inputs) - batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            # print("{}{}{}".format(inputs[excerpt], [targets[0][excerpt], targets[1][excerpt]], targets[1][excerpt]))
            yield inputs[excerpt], [targets[0][start_idx], targets[1][start_idx]]  # 每次产生batchsize个数据


def train():
    try:
        model = tf.keras.models.load_model('model')
    except OSError:
        model = c3d.mutil_output_c3d_model(scene_classes_num=2, action_classes_num=4,
                                           action_input_shape=(16, 112, 112, 3),
                                           lr=0.5)

    plot_model(model, to_file='mutil_output_c3d_model.png', show_shapes=True)

    train_dataset = get_all_dataset('train-action.list', 16, 112, 112, 3).batch(2)
    validation_dataset = get_all_dataset('test-action.list', 16, 112, 112, 3).batch(2)

    history = LossHistory()
    # 动态调整学习率
    callback = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

    model.fit(train_dataset, validation_data=validation_dataset, batch_size=2, epochs=100, callbacks=[callback,history])

    # H = model.fit_generator(minibatches(x,[y1,y2],batch_size=1,shuffle=False),
    #                         steps_per_epoch=len(x)//1,epochs=1,callbacks=[callback,history])

    history.loss_plot('epoch')
    model.save('./model/', save_format='tf')

    # x, y = get_dataset('test.list', 0, 5, 16, 112, 112, 3)
    # pre = model.predict_on_batch(x=np.array(x))
    # print(y)
    # print(pre)


train()
