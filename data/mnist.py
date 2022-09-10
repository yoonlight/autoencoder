import tensorflow as tf
from keras.datasets import mnist


def load_data():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, x_test
