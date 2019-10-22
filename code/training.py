import tensorflow as tf
from tensorflow.python.keras import regularizers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
from models import MyModel,MyDenseLayer, cal_acc
if tf.__version__ != "2.0.0":
    tf.enable_eager_execution()
def my_loss(y_pred,y_true):
    y_true = tf.one_hot(y_true, n_outputs, dtype=tf.float32)
    return  tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true,y_pred))
"""
    Hyper-parameters
"""
n_inputs = 28*28
n_outputs = 10
n_hiddens = 200
n_epochs = 1000
n_batches = 1000
epoch = 0
target_loss = 1e-5
thresh_hold = 1e-4
"""
    Load data
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train),28*28))
x_test = x_test.reshape((len(x_test),28*28))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(x_train)).batch(n_batches)
"""
    Train data
"""
m_model = MyModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
while (epoch < n_epochs) or (loss < target_loss):
    epoch += 1
    epoch_loss_avg = tf.keras.metrics.Mean()
    accuracy = 0
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            out_sm = m_model(x)
            batch_loss = my_loss(out_sm, y)
            variables = m_model.trainable_variables
            grads = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
        epoch_loss_avg(batch_loss)
    if epoch%1 == 0:
        print(m_model.summary())
    test_out = m_model(x_test)
    accuracy = cal_acc(test_out, y_test)
    m_model.sparsify_weights()
    m_model.add_layer()
    print('Epoch : {} ----- Loss : {} ----- Acc : {}'.format(epoch, epoch_loss_avg.result(), accuracy))