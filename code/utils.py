import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def get_data(params):
    """
    get dataset
    :param dataset: one of [cifar10,cifar100,mnist]
    :return: corresponding dataset
    """
    if params.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif params.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif params.dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    if params.dataset == "mnist" and params.layer_type == "dense":
        x_train = tf.reshape(x_train, (len(x_train), 28 * 28))
        x_test = tf.reshape(x_test, (len(x_test), 28 * 28))
        params.n_inputs = 28 * 28
        params.n_outputs = 10
        params.n_hiddens = 784
    elif params.dataset in ["cifar10", "cifar100"] and params.layer_type == "dense":
        x_train = tf.reshape(x_train, (x_train.shape[0], 32 * 32 * 3))
        x_test = tf.reshape(x_test, (x_test.shape[0], 32 * 32 * 3))
        params.n_inputs = 32 * 32 * 3
        params.n_outputs = 10
        params.n_hiddens = 32 * 32 * 3
    elif params.dataset in ["cifar10", "mnist"] and params.layer_type == "cnn":
        params.n_outputs = 10
        params.inp_shape = x_train.shape[1:]
    elif params.dataset == "cifar100" and params.layer_type == "cnn":
        params.n_outputs = 100
        params.inp_shape = x_train.shape[1:]
    x_train = tf.cast(x_train,tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_train = tf.cast(y_train, tf.int32)
    y_test = tf.cast(y_test, tf.int32)
    return x_train,y_train,x_test,y_test,params


