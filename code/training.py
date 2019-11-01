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
    n_layers = 8
    n_epochs = 200
    n_batches = 1000
    target_loss = 1e-5
    thresh_hold = 1e-3
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
    intializer = "zeros"
    residual = True

def run(params, test = False):
    """
        Load data
    """
    tf.random.set_seed(params.seed)
    x_train, y_train, x_test, y_test, params = get_data(params)

    full_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
        8192, seed=params.seed).batch(params.n_batches)
    train_size = int(0.8 * len(x_train))
    train_dataset = full_dataset.take(train_size).batch(params.n_batches)
    val_dataset = full_dataset.skip(train_size).batch(params.n_batches)
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
                         params.residual,
                         params.regularizer,
                         params.intializer)
    optimizer = tf.keras.optimizers.Adam()
    objective = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    """
        Logs - tensorboard setting
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/'  + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
    opt_reset = tf.group([v.initializer for v in optimizer.variables()])
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_train_acc_avg = tf.keras.metrics.Mean()
    epoch_val_acc_avg = tf.keras.metrics.Mean()
    if not os.path.exists(train_log_dir):
        os.makedirs(train_summary_writer)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_summary_writer)
    """
        Train + evaluation tf functions
    """

    @tf.function
    def train_step(images = None, labels = None, metric = metric, loss_function = objective):
        with tf.GradientTape() as tape:
            out = model(images)
            batch_loss = loss_function(labels,out)
        variables = model.trainable_variables
        grads = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        metric.update_state(labels,out)

        return batch_loss


    @tf.function
    def evaluation_step(images = None, labels = None, metric = metric, loss_function = objective):
        out = model(images)
        batch_loss = loss_function(labels,out)
        metric.update_state(labels,out)

        return batch_loss
    """
        Train data
    """
    epoch = 0
    prev_loss = 0
    prev_val_loss = 1e4
    add_layer_epoch = dict()
    while (epoch < params.n_epochs) and (epoch_train_loss_avg.result() < params.target_loss):
        epoch += 1
        # Train
        if epoch == 1:
            model.add_layer(freeze=True)
            add_layer_epoch[model.num_layers] = epoch
            print("Number of layer : {}".format(model.num_layers))
            print(model.name)
        metric.reset_states()
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                batch_loss = train_step(x,y)
            epoch_train_acc_avg(metric.result())
            epoch_train_loss_avg(batch_loss)
        #tensorboard writer - loss - weights histograms - gradient histograms
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_train_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_train_acc_avg.result(), step=epoch)
            for w in model.weights:
                tf.summary.histogram(w.name, w, step=epoch)
            # for gradient, variable in zip(grads,variables):
            #     tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient),step = epoch)
        # Validate
        metric.reset_states()
        for x, y in val_dataset:
            val_loss = evaluation_step(x,y)
            epoch_val_loss_avg(val_loss)
            epoch_val_acc_avg(metric.result())
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_val_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_val_acc_avg.result(), step=epoch)
        # post action
        # model.sparsify_weights(params.thresh_hold)
        # Add layer depending on how the train + validate losses progress
        decay_coef = 0.01/float(model.num_layers)**2
        cond_1 = abs(prev_loss - epoch_train_loss_avg.result().numpy()) < decay_coef*epoch_train_loss_avg.result().numpy()
        cond_2 = abs(prev_val_loss - epoch_val_loss_avg.result().numpy()) < decay_coef*epoch_val_loss_avg.result().numpy()
        cond_3 = epoch_train_acc_avg.result() - epoch_val_acc_avg.result() > 0.01

        if cond_1 or cond_2:
            model.add_layer(freeze=True, add = True)
            add_layer_epoch[model.num_layers] = epoch
            print("Number of layer : {} and Decay Coef : {}".format(model.num_layers,decay_coef*100))
            opt_reset
        if cond_3:
            print("Overfitting condition")
            model.use_batchnorm = True
            model.use_dropout = True
            model.update_regularizer()
            model.sparsify_weights(params.thresh_hold)
        print('Epoch : {} | Train loss : {:.3f} | Train acc : {:.3f} | Test loss : {:.3f} | Test acc : {:.3f}'.format(
            epoch,
            epoch_train_loss_avg.result(),
            epoch_train_acc_avg.result(),
            epoch_val_loss_avg.result(),
            epoch_val_acc_avg.result()))
        prev_loss = float(epoch_train_loss_avg.result().numpy())
        prev_val_loss = float(epoch_val_loss_avg.result().numpy())
        epoch_train_loss_avg.reset_states()
        epoch_val_loss_avg.reset_states()
        epoch_train_acc_avg.reset_states()
        epoch_val_acc_avg.reset_states()
    #Finish training, testing time!
    if test:
        epoch_test_loss_avg = tf.keras.metrics.Mean()
        epoch_test_acc_avg = tf.keras.metrics.Mean()
        metric.reset_states()
        for x, y in test_dataset:
            test_loss = evaluation_step(x,y)
            epoch_test_loss_avg(metric.result)
            epoch_test_acc_avg(test_loss)
        return params,model, epoch_test_loss_avg.result(),epoch_test_acc_avg.result()
    else:
        return params,model
if __name__ == "__main__":
    params = HyperParameters
    results = run(params)