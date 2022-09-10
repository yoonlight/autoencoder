from data import mnist
import tensorflow as tf


def load_data():
    x_train, x_test = mnist.load_data()
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * \
        tf.random.normal(shape=x_train.shape)
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

    x_train_noisy = tf.clip_by_value(
        x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(
        x_test_noisy, clip_value_min=0., clip_value_max=1.)

    return x_train, x_test
