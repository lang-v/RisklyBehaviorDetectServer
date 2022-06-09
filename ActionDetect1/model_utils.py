import os
import tensorflow as tf
from tensorflow import keras


def save_model(checkpoint_path, save_path):
    model = create_model()
    model.load_weights(checkpoint_path)
    model.summary()
    model.save(save_path)
    return model


def load_model(model_path):
    model = create_model()


# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


save_model('check_point/train.ckpt-34', "save_model/")
