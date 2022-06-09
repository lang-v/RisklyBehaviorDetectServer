import numpy as np

from preprocessing import *
from model import *

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
batch_size = 5
classes_num = 5
if __name__ == '__main__':
    # model = C3Dnet(classes_num, (16, 112, 112, 3))
    model = C3D_model(classes_num, (16, 112, 112, 3))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

    epochs = 100
    length = len(list(open('train.list')))
    test_length = len(list(open('test.list')))

    for epoch in range(epochs):
        start_time = time.time()

        loss = 0
        test_loss = 0
        acc = 0
        test_acc = 0

        batch_nums = length // batch_size
        test_batch_nums = test_length // batch_size

        for batch_index in range(batch_nums):
            images, labels = get_dataset('train.list', batch_index, batch_size, 16, 112, 112, 3)
            metrics = model.train_on_batch(x=np.array(images), y=np.array(labels))
            loss += metrics[0]
            acc += metrics[1]

        loss = loss / batch_nums
        acc = acc / batch_nums

        for batch_index in range(test_batch_nums):
            test_images, test_labels = get_dataset('test.list', batch_index, batch_size, 16, 112, 112, 3)
            test_metrics = model.test_on_batch(x=np.array(test_images), y=np.array(test_labels))
            test_loss += test_metrics[0]
            test_acc += test_metrics[1]

        test_loss = test_loss / test_batch_nums
        test_acc = test_acc / test_batch_nums
        print(
            "epoch {}, time:{} loss:{},acc:{}; test loss:{},test acc:{}".format(epoch, time.time() - start_time,
                                                                                loss, acc,
                                                                                test_loss, test_acc))
        if epoch%batch_size == 0:
            model.save('train_on_batch_model/', save_format='tf')

    all_dataset = get_all_dataset('test.list', 16, 112, 112, 3)
    model.evaluate(all_dataset, batch_size=batch_size)
    model.save('train_on_batch_model/', save_format='tf')
