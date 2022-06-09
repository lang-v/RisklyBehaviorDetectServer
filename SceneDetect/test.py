import os
import sys
import time

import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2 as cv


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def test():
    model = keras.models.load_model(filepath='./model/')

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './test/',
        label_mode='int',
        color_mode='rgb',
        validation_split=0.9,
        subset='validation',
        seed=122,
        image_size=(360, 480),
        batch_size=1
    )
    prediction = model.predict(x=val_ds, batch_size=1)

    for p in prediction:
        print("balcony:{} flat:{} roof:{}".format((p[0] * 100).__format__(".2f"), ((p[1] * 100).__format__(".2f")),
                                            ((p[2] * 100).__format__(".2f"))))
    result = model.evaluate(x=val_ds, batch_size=1)
    print(result)


def write_graph_to_pb():
    m = tf.saved_model.load('./model/')

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    tfm = tf.function(lambda x: m(x))  # full model
    tfm = tfm.get_concrete_function(tf.TensorSpec(m.signatures['serving_default'].inputs[0].shape.as_list(),
                                                  m.signatures['serving_default'].inputs[0].dtype.name))
    frozen_func = convert_variables_to_constants_v2(tfm)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./model/", name="saved_model_graph.pb", as_text=False)


def camera():
    width = 360
    height = 360

    net = keras.models.load_model("./model/")
    # cv.dnn.readNet()
    # net = cv.dnn.readNetFromTensorflow("./model/saved_model_graph.pb")

    # cap = cv.VideoCapture('outside-test1.mp4')
    cap = cv.VideoCapture('test/下楼梯.mp4')

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            cv.waitKey()
            break

        # blob = cv.dnn.blobFromImage(frame, 1.0/255, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=True)
        # net.setInput(blob)
        # out = net.forward()

        image = cv.resize(frame,(height,width))
        arr = tf.keras.preprocessing.image.img_to_array(image)
        arr = np.expand_dims(arr,0)
        # dataset = tf.data.Dataset.from_tensor_slices([image])
        # dataset = dataset.batch(batch_size=1)

        s = time.time_ns()
        out = net.predict(arr, batch_size=1)
        t = time.time_ns()
        mt = (t-s)/1000000
        # print("C2D 单帧检测耗时:{}ms FPS:{}".format(mt,1000/mt))

        text = get_text(out)
        cv.putText(frame, text, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0xFF, 0xFF, 0xFF))
        cv.imshow('Scene Detect', frame)


classes = ['indoors','outside','rooftop']

def get_text(out):
    out = out[0]
    index = np.argmax(out)
    value = out[index]
    text = classes[index]
    text += ":{}  ".format((value*100).__format__(".2f"))
    # print(str(out))
    print(text)
    return text


# test()
camera()
# write_graph_to_pb()
