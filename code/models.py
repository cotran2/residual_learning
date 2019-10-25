import tensorflow as tf
from tensorflow.python.keras import regularizers,initializers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
from tensorflow.keras import layers

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, shape = None, layer_name = None , kernel_regularizer = None, bias_regularizer = None, initializer = 'zeros'):
    super(MyDenseLayer, self).__init__(name = layer_name)
    self.num_outputs = num_outputs
    self.layer_name = layer_name
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.shape = shape
    self.kernel_initializer = initializers.get(initializer)
    self.bias_initializer = initializers.get(initializer)
  def build(self,input_shape):
    self.kernel = self.add_weight("kernel_{}".format(self.layer_name),
                                    shape=[int(self.shape[-1]),
                                           self.num_outputs],
                                 regularizer=self.kernel_regularizer,
                                 initializer = self.kernel_initializer)
    self.bias = self.add_weight("bias_{}".format(self.layer_name),
                                    shape=[self.num_outputs],
                                 regularizer=self.bias_regularizer,
                                initializer = 'zeros')
  def call(self, input, activation_function = tf.nn.relu):
    if activation_function:
        return activation_function(tf.add(tf.matmul(input, self.kernel), self.bias))
    else:
        return tf.add(tf.matmul(input, self.kernel), self.bias)

class DenseModel(tf.keras.Model):
  def __init__(self,n_hiddens = 200,n_inputs = 784,n_outputs = 10):
    super(DenseModel, self).__init__()
    self.n_hiddens = n_hiddens
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.num_layers = 1
    self.input_layer = MyDenseLayer(n_hiddens,shape = (None,self.n_inputs),
                                    layer_name ='input',
                                    initializer = "RandomNormal"
                                    )
    self.list_dense = [self.input_layer]
    self.output_layer = MyDenseLayer(
        n_outputs,shape = (None,self.n_hiddens),
        layer_name = 'output',
        initializer = "RandomNormal")
  def call(self, inputs):
    for index,layer in enumerate(self.list_dense):
      if index == 0:
        if self.n_hiddens == self.n_inputs:
            out = inputs + layer(inputs)
        else:
            out = layer(inputs)
      elif 1<=index <= self.num_layers-1:
        prev_out = out
        out = prev_out + layer(out)
    out = self.output_layer(out,activation_function = None)
    return out
  def add_layer(self):
    self.num_layers += 1
    new_dense = MyDenseLayer(self.n_hiddens,
                             shape = (None,self.n_hiddens),
                             layer_name =str(self.num_layers)
                             )
    self.list_dense.append(new_dense)
    for index in range(len(self.layers)-2):
      self.layers[index].trainable = False
  def sparsify_weights(self, threshold = 1e-6):
    weights = self.list_dense[-1].get_weights()
    sparsified_weights = []
    for w in weights:
        bool_mask = (w > threshold).astype(int)
        sparsified_weights.append(w*bool_mask)
    self.list_dense[-1].set_weights(sparsified_weights)
class CNNModel(tf.keras.Model):
  def __init__(self, n_filters = 64,n_kernels = 3,n_outputs = 10, inp_shape = (28,28),regularizer = None):
    """
    Adaptive layer-wise training model
    :param n_filters: number of filters
    :param n_kernels: kernels size
    :param n_outputs: number of output classes
    :param inp_shape: dimension of the inputs
    """
    super(CNNModel, self).__init__()
    self.conv_dim = len(inp_shape)-1
    self.n_filters = n_filters
    self.n_kernels = n_kernels
    self.n_outputs = n_outputs
    self.num_layers = 1
    self.inp_shape = inp_shape
    self.regularizer = regularizer
    if self.conv_dim == 1:
        self.input_layer = layers.Conv1D(self.n_filters, (self.n_kernels),
                                         activation = "linear",
                                         input_shape = self.inp_shape,
                                         padding = "same",
                                         name ='cnn_input',
                                         kernel_regularizer = self.regularizer,
                                         bias_regularizer = self.regularizer
                                        )
        self.output_layer = layers.Conv1D(self.n_kernels, (self.n_kernels),
                                         activation="linear",
                                         input_shape=(None, self.inp_shape[0], self.n_filters),
                                         padding="same",
                                         name='cnn_input',
                                         kernel_regularizer=self.regularizer,
                                         bias_regularizer=self.regularizer
                                         )
    elif self.conv_dim == 2:
        self.input_layer = layers.Conv2D(self.n_filters, (self.n_kernels,self.n_kernels),
                                         activation="linear",
                                         input_shape=self.inp_shape,
                                         padding = "same",
                                         name='cnn_input',
                                         kernel_regularizer=self.regularizer,
                                         bias_regularizer=self.regularizer
                                         )
        self.output_layer = layers.Conv2D(self.n_kernels, (self.n_kernels, self.n_kernels),
                                         activation= "linear",
                                         input_shape=(None, self.inp_shape[0],self.inp_shape[1], self.n_filters),
                                         padding="same",
                                         name="cnn_output",
                                         kernel_regularizer=self.regularizer,
                                         bias_regularizer=self.regularizer
                                         )
    self.list_cnn = [self.input_layer]
    self.flatten = layers.Flatten()

    #compute input shape after flatten for the dense layer
    self.class_inp = np.prod(self.inp_shape[:-1])*self.n_kernels
    self.classify = MyDenseLayer(
        self.n_outputs,shape = (None,self.class_inp),
        layer_name = 'classify',
        initializer = "RandomNormal")
  def call(self, inputs):
    for index,layer in enumerate(self.list_cnn):
      if index == 0:
        out = layer(inputs)
      else:
        prev_out = out
        out = prev_out + layer(out)
    out = self.output_layer(out)
    out = self.flatten(out)
    out = self.classify(out,activation_function = None)
    return out
  def add_layer(self):
    """
    add an layer to the model
    :return:
    """
    self.num_layers += 1
    if self.conv_dim == 1:
        new_cnn = layers.Conv1D(self.n_filters,
                                (self.n_kernels),
                                activation='relu',
                                input_shape=(None, self.inp_shape[0], self.n_filters),
                                padding="same",
                                name='cnn_1d_{}'.format(self.num_layers-1),
                                kernel_initializer = initializers.get("zeros"),
                                bias_initializer=initializers.get("zeros"),
                                kernel_regularizer=self.regularizer,
                                bias_regularizer=self.regularizer
                                )
    elif self.conv_dim == 2:
        new_cnn = layers.Conv2D(self.n_filters,
                                (self.n_kernels, self.n_kernels),
                                activation='relu',
                                input_shape=(None, self.inp_shape[0],self.inp_shape[1], self.n_filters),
                                padding="same",
                                name='cnn_2d_{}'.format(self.num_layers-1),
                                kernel_initializer=initializers.get("zeros"),
                                bias_initializer=initializers.get("zeros"),
                                kernel_regularizer=self.regularizer,
                                bias_regularizer=self.regularizer
                                )
    self.list_cnn.append(new_cnn)
    for index in range(1,self.num_layers-1):
      self.list_cnn[index].trainable = False
  def sparsify_weights(self, threshold = 1e-6):
    """
    sparsify the last added cnn layer
    :param threshold: if weight < threshold -> set weight = 0
    :return:
    """
    weights = self.list_cnn[-1].get_weights()
    sparsified_weights = []
    for w in weights:
        bool_mask = (w > threshold).astype(int)
        sparsified_weights.append(w*bool_mask)
    self.list_cnn[-1].set_weights(sparsified_weights)

def cal_acc(y_pred,y_true):
    """
    Calculate accuracy
    :param y_pred: softmax output of the model
    :param y_true: targets
    :return: accuracy
    """
    correct = tf.math.in_top_k(tf.cast(tf.squeeze(y_true),tf.int64),tf.cast(y_pred, tf.float32),  1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
def my_loss(y_pred,y_true,n_outputs):
    """
    Calculate cross entropy loss
    :param y_pred: logits output from the model
    :param y_true: targets
    :param n_outputs: number of classes
    :return: loss
    """
    y_true = tf.one_hot(tf.cast(y_true,tf.int64), n_outputs, dtype=tf.float32)
    return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred))