import os
import threading
from threading import Thread
import time

import cv2 as cv
import tensorflow as tf
import numpy as np
import queue
import preprocessing

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

classes = ['fight', 'throw', 'walk']

FRAME_SIZE = 20


class FrameQueue(object):

    def __init__(self, max_size=FRAME_SIZE):
        self.max_size = max_size
        self.frames = list()
        self.current_index = 0

    def pop(self):
        if self.current_index == 0:
            return None
        self.current_index -= 1
        result = self.frames.pop()

        return result

    def put(self, frame):
        if len(self.frames) == self.max_size:
            # 移除最后那个元素，从index=0处重新开始入队，保证数据流动
            self.frames.pop()
            # self.current_index = 0
        elif len(self.frames) < self.max_size:
            self.frames.append(frame)
        else:
            self.frames[self.current_index] = frame
        self.current_index += 1

    def full(self):
        return len(self.frames) == self.max_size

    def all(self):
        return self.frames


def main():
    frame_queue = FrameQueue()
    # result_queue = queue.Queue(maxsize=1)

    # compute_thread = Thread(target=compute_action, args=(frame_queue, result_queue))
    # compute_thread.start()
    model = tf.keras.models.load_model("model_50epochs_crop_valacc96_valloss35")

    cap = cv.VideoCapture("D:\数据集备份\dataset_source/fight/fight3.mp4")
    # cap = cv.VideoCapture(0)

    time_split = 190
    t = time.time()
    t = int(round(t * 1000))
    text = 'type:?'
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        while int(round(time.time() * 1000)) - t < time_split:
            continue
        t = int(round(time.time() * 1000))
        if not has_frame:
            cv.waitKey()
            cap.release()
            break
        # 下面的操作保证整个队列中的帧 是流动的
        # if frame_queue.full():
        #     frame_queue.get()
        image = cv.resize(frame, (112, 112))
        frame_queue.put(image)

        # text = 'type:?'
        if frame_queue.full():
            frames = frame_queue.all()
            # for i in range(16):
            #     frames.append(frame_queue.get())
            text = real_compute(model, frames)
            print(text)

        # print("FPS:{}".format(cap.get(cv.CAP_PROP_FPS)))
        cv.putText(frame, text, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0xFF, 0xFF, 0xFF))
        cv.imshow("action classifier", frame)
        time.sleep(0.17)
        # cv.waitKey()


# q 就是帧队列
def compute_action(frame_queue, result_queue):
    model = tf.keras.models.load_model("model_50epochs_crop_valacc96_valloss35")
    model.summary()
    while True:
        if not frame_queue.full():
            time.sleep(0.1)
            continue
        # 全部取出来 或许有更好的实现办法
        frames = []
        for i in range(16):
            frames.append(tf.keras.preprocessing.image.img_to_array(frame_queue.get()))
        frames = np.array(frames) / 255. + 1e-5

        batch = np.reshape(frames, newshape=(1, FRAME_SIZE, 112, 112, 3))
        arr = model.predict_on_batch(batch)

        print(arr)

        max_index = np.argmax(arr[0])
        res = classes[max_index] + ":{:.2}%".format(arr[0][max_index] * 100)
        if result_queue.not_empty:
            result_queue.get()
        result_queue.put(res)
        print(res)


def real_compute(model, frames):
    frames = np.array(frames) / 255. + 1e-5
    batch = np.reshape(frames, newshape=(1, FRAME_SIZE, 112, 112, 3))
    arr = model.predict_on_batch(batch)
    max_index = np.argmax(arr[0])
    res = classes[max_index] + ":{:.2f}%".format(arr[0][max_index] * 100)
    return res


if __name__ == '__main__':
    main()
