import tensorflow as tf
from tensorflow.python.keras import regularizers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
from utils import *
from models import *

if tf.__version__ != "2.0.0":
    tf.enable_eager_execution()
"""
    Hyper-parameters
"""


class HyperParameters:
    """
    Easy to manage HyperParameters
    """
    n_epochs = 20
    n_batches = 1000
    target_loss = 1e-5
    thresh_hold = 1e-5
    dataset = "mnist"
    regularizer = regularizers.l1(0.01)
    layer_type = "cnn"
    seed = 1234
    n_inputs = 28 * 28
    n_outputs = 10
    n_hiddens = 784
    n_filters = 64
    n_kernels = 3
    n_outputs = 100
    inp_shape = None

if __name__ == "__main__":
    params = HyperParameters
    """
        Load data
    """
    tf.random.set_seed(params.seed)
    x_train, y_train, x_test, y_test, params = get_data(params)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.n_batches)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(
        8192, seed = params.seed).batch(params.n_batches)
    """
        Set model
    """
    if params.layer_type == "dense":
        model = DenseModel(params.n_hiddens, params.n_inputs, params.n_outputs)
    elif params.layer_type == "cnn":
        model = CNNModel(params.n_filters, params.n_kernels, params.n_outputs, params.inp_shape , params.regularizer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    """
        Train data
    """
    epoch = 0
    while (epoch < params.n_epochs) or (epoch_loss_avg.result() < params.target_loss):
        epoch += 1
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_train_acc_avg = tf.keras.metrics.Mean()
        epoch_test_acc_avg = tf.keras.metrics.Mean()
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                out = model(x)
                batch_loss = my_loss(out, y, params.n_outputs)
                variables = model.trainable_variables
                grads = tape.gradient(batch_loss, variables)
                optimizer.apply_gradients(zip(grads, variables))
            epoch_train_acc_avg(cal_acc(out, y))
            epoch_loss_avg(batch_loss)
        if epoch % 1 == 0:
            print(model.summary())
        for x, y in test_dataset:
            test_out = model(x)
            epoch_test_acc_avg(cal_acc(test_out, y))
        model.sparsify_weights(params.thresh_hold)
        model.add_layer()
        print('Epoch : {} ----- Loss : {:.4f} ----- Train acc : {:.4f} ----- Test Acc : {:.4f}'.format(
            epoch, epoch_loss_avg.result(), epoch_train_acc_avg.result(), epoch_test_acc_avg.result()))