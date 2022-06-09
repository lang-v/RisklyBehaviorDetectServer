import io
import os
import time
# from random import Random
import random

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import layers


# 获取全部数据 @return Dataset
def get_all_dataset(list_path, clip_size, width, height, channel_num):
    lines = open(list_path, 'r')
    lines = list(lines)
    random.shuffle(lines)
    clips = []
    labels = []
    video_indices = range(len(lines))
    # 每次都从上次加载的地方开始读
    for i in video_indices:
        line = lines[i].strip('\n').split()
        path = line[0]
        clip = generate_clip(path, clip_size, width, height, channel_num)
        clips.append(clip)
        labels.append(int(line[1]))

    # all_range = list(range(len(lines)))
    # random.shuffle(all_range)
    # train_range = all_range[:int(len(all_range)) * 0.8]
    # validation_range = all_range[int(len(all_range)) * 0.8:]

    labels = LabelBinarizer().fit_transform(np.array(labels))

    # return tf.data.Dataset.from_tensor_slices((clips,labels))
    return tf.data.Dataset.from_tensor_slices(
        (clips[:int(len(lines) * 0.8)], labels[:int(len(lines) * 0.8)])), tf.data.Dataset.from_tensor_slices(
        (clips[int(len(lines) * 0.8):], labels[int(len(lines) * 0.8):]))


# 分段获取数据集 @return images,labels
def get_dataset(list_path, batch_index, batch_size, clip_size, width, height, channel_num):
    lines = open(list_path, 'r')
    lines = list(lines)
    random.shuffle(lines)
    clips = []
    labels = []
    video_indices = range(len(lines))
    # 每次都从上次加载的地方开始读
    for i in video_indices[batch_index * batch_size:(batch_index + 1) * batch_size]:
        line = lines[i].strip('\n').split()
        path = line[0]
        clip = generate_clip(path, clip_size, width, height, channel_num)
        clips.append(clip)
        labels.append(int(line[1]))

    return clips, labels
    # return tf.data.Dataset.from_tensor_slices((clips, labels)).batch(batch_size)


# 负责将单个视频的多帧图像组装起来
def generate_clip(filepath, clip_size, width, height, channel_num):
    files = os.listdir(filepath)
    arr = []

    for i in range(clip_size):
        tmp_img = keras.preprocessing.image.load_img(filepath + '\\' + files[i % len(files)],
                                                     target_size=(width, height),
                                                     color_mode='rgb')
        arr.append(keras.preprocessing.image.img_to_array(tmp_img))
    clip = np.array(arr) / 255.0 + 1e-5
    return clip
