import io
import os
import time

from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import keras as keras
import numpy as np
from tensorflow.python.keras import layers



# 获取全部数据 @return Dataset
def get_all_dataset(list_path, clip_size, width, height, channel_num):
    lines = open(list_path, 'r')
    lines = list(lines)
    clips = []
    labels = []
    labels_2 = []
    video_indices = range(len(lines))
    # 每次都从上次加载的地方开始读
    for i in video_indices:
        line = lines[i].strip('\n').split()
        path = line[0]
        clip = generate_clip(path, clip_size, width, height, channel_num)
        clips.append(clip)
        labels.append(int(line[1]))
        labels_2.append(int(line[2]))

    # outputs = np.array([labels,labels_2]).reshape((43,1,2))
    # o1 = np.array(labels).reshape((len(labels),1))
    # o2 = np.array(labels_2).reshape((len(labels_2),1))

    #  当二分类时,出现shape=(None,1)
    o1 = LabelBinarizer().fit_transform(labels)
    o2 = LabelBinarizer().fit_transform(labels_2)
    return tf.data.Dataset.from_tensor_slices((clips, {"fc_scene":o1,"fc_action":o2}))
    # y1 = np.array(labels)
    # y2 = np.array(labels_2)
    # clips = np.array(clips)
    # return clips, tf.data.Dataset.from_tensor_slices(([labels,labels_2]))
    # return clips, y1, y2


# 分段获取数据集 @return images,labels
def get_dataset(list_path, batch_index, batch_size, clip_size, width, height, channel_num):
    lines = open(list_path, 'r')
    lines = list(lines)
    clips = []
    labels = []
    labels_2 = []
    video_indices = range(len(lines))
    # 每次都从上次加载的地方开始读
    for i in video_indices[batch_index * batch_size:(batch_index + 1) * batch_size]:
        line = lines[i].strip('\n').split()
        path = line[0]
        clip = generate_clip(path, clip_size, width, height, channel_num)
        clips.append(clip)
        labels.append(int(line[1]))
        labels_2.append(int(line[2]))

    return clips, [labels, labels_2]
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
    clip = np.array(arr)
    return clip
