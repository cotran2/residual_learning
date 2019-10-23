import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def get_data(dataset):
    """
    get dataset
    :param dataset: one of [cifar10,cifar100,mnist]
    :return: corresponding dataset
    """
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.cast(x_train,tf.float32)
    x_test = tf.cast(x_test, tf.int32)
    y_train = tf.cast(y_train, tf.float32)
    y_test = tf.cast(y_test, tf.int32)
    return x_train,y_train,x_test,y_test


