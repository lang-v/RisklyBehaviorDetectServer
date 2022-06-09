import io
import os
import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, Input
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.applications.densenet import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, ZeroPadding3D
import model
from preprocessing import *


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.keras.backend.clear_session()

IMAGE_W = 112
IMAGE_H = 112

CHANNEL_NUM = 3
FRAME_SIZE = 16  # 每个视频截取多少帧出来，如果取少了就会导致正确率很低
BATCH_SIZE = 5
CLASSES_NUM = 5
EPOCHS = 100




strategy = tf.distribute.MirroredStrategy()

# dataset = get_all_dataset('./train.list', FRAME_SIZE, 112, 112, 3)
# train_dataset = dataset.shuffle(buffer_size=dataset.__len__()).batch(BATCH_SIZE)
# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
#
# dataset = get_all_dataset('./test.list', FRAME_SIZE, 112, 112, 3)
# val_dataset = dataset.batch(BATCH_SIZE)
# val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)


def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)


with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')


with strategy.scope():
    model = None
    try:
        model = keras.models.load_model('model/')
    except OSError:
        # model = C3D_model()
        model = model.C3Dnet(CLASSES_NUM, input_shape=(FRAME_SIZE, IMAGE_W, IMAGE_H, CHANNEL_NUM))

    optimizer = tf.keras.optimizers.Adam()


def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss


def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))

    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)


@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))


for epoch in range(EPOCHS):
    start_time = time.time()

    # val_dataset = get_dataset('./test.list', epoch, BATCH_SIZE, FRAME_SIZE, 112, 112, 3)
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    length = len(list(open('train.list')))
    for i in range(length // BATCH_SIZE):
        train_dataset = get_dataset('./train.list', i,  BATCH_SIZE, FRAME_SIZE, 112, 112, 3)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        for inputs in train_dist_dataset:
            total_loss += distributed_train_step(inputs)
            tf.keras.backend.clear_session()
        num_batches += 1
    train_loss = total_loss / num_batches

    # TEST LOOP
    for i in range(length // BATCH_SIZE):
        val_dataset = get_dataset('./test.list', i, BATCH_SIZE, FRAME_SIZE, 112, 112, 3)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        val_dataset = val_dataset.with_options(options)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
        for inputs in val_dist_dataset:
            distributed_test_step(inputs)
            tf.keras.backend.clear_session()

    if epoch % 2 == 0:
        model.save("./model/", 'tf')

    template = ("Epoch {},time: {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                "Test Accuracy: {}")
    print(template.format(epoch + 1, time.time()-start_time, train_loss,
                          train_accuracy.result() * 100, test_loss.result(),
                          test_accuracy.result() * 100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
