from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import os
import tensorflow as tf
from ActionDetect.preprocessing import get_all_dataset, get_dataset
from model import C3D_model, C3Dnet
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import matplotlib.pyplot as plt


# 写一个LossHistory类，保存loss和acc
class LossHistory(tf.keras.callbacks.Callback):
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


# model = tf.keras.models.load_model("./train_on_batch_model")
try:
    model = tf.keras.models.load_model('model')
except OSError:
    model = C3D_model(3, input_shape=(20, 112, 112, 3))
    # 这里使用Adam效果极差20epochs val_loss val_acc 都不变化
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])

# save model graph to local with .png
tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True)

# batch_size = 2 # loss down acc up; val_loss concussion val_acc up;
# 训练集正确率已经接近1了， 但是验证集 正确率卡在0.6829偶尔震荡，val_loss 上升到2.5 左右继续震荡, 过拟合 扩充数据集
#
# batch_size = 5 # 数据集容量翻倍后，训练50次得到了不错的结果，但是感觉还有优化空间

batch_size = 2

train_all_dataset, validation_all_dataset = get_all_dataset('train.list', 20, 112, 112, 3)#, \
                                            # get_all_dataset('test.list', 16, 112, 112, 3)

train_all_dataset = train_all_dataset.batch(batch_size)
validation_all_dataset = validation_all_dataset.batch(batch_size)

history = LossHistory()

callback = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
model.fit(train_all_dataset, validation_data=validation_all_dataset, batch_size=batch_size, epochs=50,
          callbacks=[callback, history])
history.loss_plot('epoch')

model.save('./model_50epochs_crop', save_format='tf')

# x, y = get_dataset('test.list', 0, 5, 16, 112, 112, 3)
# pre = model.predict_on_batch(x=np.array(x))
# print(y)
# print(pre)
