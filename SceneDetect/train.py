import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.data import AUTOTUNE

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

CLASSES_NUM = 3
IMAGE_W = 360
IMAGE_H = 360
BATCH_SIZE = 5

TRAIN_PATH = './dataset/'


def generate_model():
    model = keras.Sequential([
        # layers.Rescaling(1. / 255, input_shape=(IMAGE_W, IMAGE_H, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMAGE_W, IMAGE_H, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # layers.Dense(3),
        layers.Dense(units=CLASSES_NUM, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.outputs)
    return model


def get_batch_pro():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        label_mode='int',
        color_mode='rgb',
        subset='training',
        seed=122,
        shuffle=True,
        validation_split=0.2,
        image_size=(IMAGE_W, IMAGE_H),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        label_mode='int',
        color_mode='rgb',
        subset='validation',
        validation_split=0.2,
        shuffle=True,
        seed=122,
        image_size=(IMAGE_W, IMAGE_H),
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds


def train():
    train_ds, val_ds = get_batch_pro()
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    model = generate_model()
    plot_model(model, "scene.png", show_shapes=True)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    hist = model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE, epochs=5)
    training_vis(hist)

    model.save("model")


# define the function
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']  # new version => hist.history['accuracy']
    val_acc = hist.history['val_accuracy']  # => hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()


train()
