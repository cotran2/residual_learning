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
    layer_type = "dense"
    seed = 1234

params = HyperParameters

"""
    Load data
"""
tf.random.set_seed(params.seed)
x_train, y_train, x_test, y_test= get_dataset(dataset)
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:]
if params.dataset=="mnist" and params.layer_type == "dense":
    x_train = x_train.reshape((len(x_train),28*28))
    x_test = x_test.reshape((len(x_test),28*28))
    n_inputs = 28*28
    n_outputs = 10
    n_hiddens = 784
elif params.dataset in ["cifar10","cifar100"] and params.layer_type == "dense":
    x_train = x_train.reshape((len(x_train), 32 * 32 * 3))
    x_test = x_test.reshape((len(x_test), 32 * 32 * 3))
    n_inputs = 32*32*3
    n_outputs = 10
    n_hiddens = 32*32*3

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(
    8192, seed = params.seed).batch(params.n_batches)
"""
    Set model
"""
if params.layer_type == "dense":
    model = MyModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
"""
    Train data
"""
epoch = 0
while (epoch < n_epochs) or (epoch_loss_avg < target_loss):
    epoch += 1
    epoch_loss_avg = tf.keras.metrics.Mean()
    accuracy = 0
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            out = m_model(x)
            batch_loss = my_loss(out, y)
            variables = m_model.trainable_variables
            grads = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
        epoch_loss_avg(batch_loss)
    if epoch%1 == 0:
        print(m_model.summary())
    test_out = m_model(x_test)
    accuracy = cal_acc(test_out, y_test)
    m_model.sparsify_weights(thresh_hold)
    m_model.add_layer()
    print('Epoch : {} ----- Loss : {} ----- Acc : {}'.format(epoch, epoch_loss_avg.result(), accuracy))