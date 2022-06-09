import tensorflow as tf
import numpy as np
import C3D_model
import data_processing

TRAIN_LOG_DIR = 'Log/train/'
TRAIN_CHECK_POINT = 'check_point/train.ckpt-34'
TEST_LIST_PATH = 'test.list'
BATCH_SIZE = 10
NUM_CLASSES = 4
CROP_SZIE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16
EPOCH_NUM = 50
test_num = data_processing.get_test_num(TEST_LIST_PATH)

test_video_indices = range(test_num)


def test():
    with tf.Graph().as_default():
        batch_clips = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM],
                                               name='X')
        batch_labels = tf.compat.v1.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='Y')
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)
        accuracy = tf.reduce_mean(
            input_tensor=tf.cast(tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=batch_labels, axis=1)),
                                 np.float32))

        restorer = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            restorer.restore(sess, TRAIN_CHECK_POINT)
            accuracy_epoch = 0
            batch_index = 0
            for i in range(test_num // BATCH_SIZE):
                if i % 10 == 0:
                    print('Testing %d of %d' % (i + 1, test_num // BATCH_SIZE))
                batch_data, batch_index = data_processing.get_batches(TEST_LIST_PATH, NUM_CLASSES, batch_index,
                                                                      test_video_indices, BATCH_SIZE)
                accuracy_out = sess.run(accuracy, feed_dict={batch_clips: batch_data['clips'],
                                                             batch_labels: batch_data['labels'],
                                                             keep_prob: 1.0})
                accuracy_epoch += accuracy_out

        print('Test accuracy is %.5f' % (accuracy_epoch / (test_num // BATCH_SIZE)))
