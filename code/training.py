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
    n_epochs = 4
    n_batches = 1000
    target_loss = 1e-5
    thresh_hold = 1e-5
    dataset = "cifar10"
    layer_type = "cnn"
    seed = 1234
    if layer_type == "dense":
        n_inputs = 28 * 28
        n_outputs = 10
        n_hiddens = 784

params = HyperParameters

"""
    Load data
"""
tf.random.set_seed(params.seed)
x_train, y_train, x_test, y_test = get_data(params.dataset)
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:]
if params.dataset == "mnist" and params.layer_type == "dense":
    x_train = tf.reshape(x_train, (len(x_train), 28 * 28))
    x_test = tf.reshape(x_test, (len(x_test), 28 * 28))
    n_inputs = 28 * 28
    n_outputs = 10
    n_hiddens = 784
elif params.dataset in ["cifar10", "cifar100"] and params.layer_type == "dense":
    x_train = tf.reshape(x_train, (x_train.shape[0], 32 * 32 * 3))
    x_test = tf.reshape(x_test, (x_test.shape[0], 32 * 32 * 3))
    n_inputs = 32 * 32 * 3
    n_outputs = 10
    n_hiddens = 32 * 32 * 3
elif params.dataset in ["cifar10", "mnist"] and params.layer_type == "cnn":
    n_filters = 64
    n_kernels = 3
    n_outputs = 10
    inp_shape = x_train.shape[1:]
elif params.dataset == "cifar100" and params.layer_type == "cnn":
    n_filters = 64
    n_kernels = 3
    n_outputs = 100
    inp_shape = x_train.shape[1:]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
    8192, seed=params.seed).batch(params.n_batches)
"""
    Set model
"""
if params.layer_type == "dense":
    model = DenseModel(n_hiddens, n_inputs, n_outputs)
elif params.layer_type == "cnn":
    model = CNNModel(n_filters, n_kernels, n_outputs, inp_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
"""
    Train data
"""
epoch = 0
while (epoch < params.n_epochs) or (epoch_loss_avg.result() < params.target_loss):
    epoch += 1
    epoch_loss_avg = tf.keras.metrics.Mean()
    accuracy = 0
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            out = model(x)
            batch_loss = my_loss(out, y, n_outputs)
            variables = model.trainable_variables
            grads = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
        epoch_loss_avg(batch_loss)
    if epoch % 1 == 0:
        print(model.summary())
    # test_out = model(x_test)
    # accuracy = cal_acc(test_out, y_test)
    # print('Epoch : {} ----- Loss : {} ----- Acc : {}'.format(epoch, epoch_loss_avg.result(), accuracy))
    model.sparsify_weights(params.thresh_hold)
    model.add_layer()
