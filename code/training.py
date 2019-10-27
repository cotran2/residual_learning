import tensorflow as tf
from tensorflow.python.keras import regularizers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
import datetime
from utils import *
from models import *
import os
if tf.__version__ != "2.0.0":
    tf.enable_eager_execution()
"""
    Hyper-parameters
"""


class HyperParameters:
    """
    Easy to manage
    """
    n_epochs = 100
    n_batches = 1000
    target_loss = 1e-5
    thresh_hold = 1e-5
    dataset = "cifar10"
    regularizer = None
    layer_type = "cnn"
    seed = 1234
    n_inputs = 28 * 28
    n_outputs = 10
    n_hiddens = 784
    n_filters = 64
    n_kernels = 3
    n_outputs = 100
    inp_shape = None
    intializer = "RandomNormal"

def train(params):
    """
        Load data
    """
    tf.random.set_seed(params.seed)
    x_train, y_train, x_test, y_test, params = get_data(params)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.n_batches)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
        8192, seed=params.seed).batch(params.n_batches)
    """
        Set model and log directories
    """
    if params.layer_type == "dense":
        model = DenseModel(params.n_hiddens, params.n_inputs, params.n_outputs)
    elif params.layer_type == "cnn":
        model = CNNModel(params.n_filters,
                         params.n_kernels,
                         params.n_outputs,
                         params.inp_shape,
                         params.regularizer,
                         params.intializer)
    optimizer = tf.keras.optimizers.Adam()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/'  + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_test_loss_avg = tf.keras.metrics.Mean()
    epoch_train_acc_avg = tf.keras.metrics.Mean()
    epoch_test_acc_avg = tf.keras.metrics.Mean()
    if not os.path.exists(train_log_dir):
        os.makedirs(train_summary_writer)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_summary_writer)
    """
        Train data
    """
    epoch = 0
    while (epoch < params.n_epochs) or (epoch_train_loss_avg.result() < params.target_loss):
        epoch += 1
        # Train
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                out = model(x)
                batch_loss = my_loss(out, y, params.n_outputs)
                variables = model.trainable_variables
                grads = tape.gradient(batch_loss, variables)
                optimizer.apply_gradients(zip(grads, variables))
            epoch_train_acc_avg(cal_acc(out, y))
            epoch_train_loss_avg(batch_loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_train_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_train_acc_avg.result(), step=epoch)
            for w in model.weights:
                tf.summary.histogram(w.name, w, step=epoch)
        # Test
        for x, y in test_dataset:
            test_out = model(x)
            epoch_test_loss_avg(my_loss(test_out, y, params.n_outputs))
            epoch_test_acc_avg(cal_acc(test_out, y))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_test_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_test_acc_avg.result(), step=epoch)
        # post action
        # if epoch % 1 == 0:
        #     print(model.summary())
        # model.sparsify_weights(params.thresh_hold)
        if epoch == 1:
            model.add_layer(freeze=True)
            print("Number of layer : {}".format(model.num_layers))
        elif abs(prev_loss - epoch_train_loss_avg.result().numpy()) <= 0.03:
            model.add_layer(freeze=True, add = True)
            print("Number of layer : {}".format(model.num_layers))
        else:
            print(prev_loss, epoch_train_loss_avg.result().numpy())
            model.add_layer(freeze=False,add=False)
        print('Epoch : {} | Train loss : {:.3f} | Train acc : {:.3f} | Test loss : {:.3f} | Test acc : {:.3f}'.format(
            epoch,
            epoch_train_loss_avg.result(),
            epoch_train_acc_avg.result(),
            epoch_test_loss_avg.result(),
            epoch_test_acc_avg.result()))
        prev_loss = float(epoch_train_loss_avg.result().numpy())
        epoch_train_loss_avg.reset_states()
        epoch_test_loss_avg.reset_states()
        epoch_train_acc_avg.reset_states()
        epoch_test_acc_avg.reset_states()

if __name__ == "__main__":
    params = HyperParameters
    train(params)



