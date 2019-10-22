import tensorflow as tf
from tensorflow.python.keras import regularizers,initializers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
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

def cal_acc(y_pred,y_true):
    correct = tf.math.in_top_k(tf.cast(y_true,tf.int64),tf.cast(y_pred, tf.float32),  1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

class MyModel(tf.keras.Model):
  def __init__(self,n_hiddens = 200,n_inputs = 784,n_outputs = 10):
    super(MyModel, self).__init__()
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
    weights = self.layers[-2].get_weights()
    sparsified_weights = []
    for w in weights:
        bool_mask = (w > threshold).astype(int)
        sparsified_weights.append(w*bool_mask)
    self.layers[-2].set_weights(sparsified_weights)
