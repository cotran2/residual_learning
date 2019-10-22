import tensorflow as tf
from tensorflow.python.keras import regularizers
import pandas as pd
from tensorflow.python.ops import nn
import numpy as np
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, shape = None, layer_name = None , kernel_regularizer = None, bias_regularizer = None):
    super(MyDenseLayer, self).__init__(name = layer_name)
    self.num_outputs = num_outputs
    self.layer_name = layer_name
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.shape = shape
  def build(self,input_shape):
    self.kernel = self.add_weight("kernel_{}".format(self.layer_name),
                                    shape=[int(self.shape[-1]),
                                           self.num_outputs],
                                 regularizer=self.kernel_regularizer)
    self.bias = self.add_weight("bias_{}".format(self.layer_name),
                                    shape=[self.num_outputs],
                                 regularizer=self.bias_regularizer)
  def call(self, input):
    return nn.bias_add(tf.matmul(input, self.kernel),self.bias)

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
    self.input_layer = MyDenseLayer(n_hiddens,shape = (None,self.n_inputs),layer_name ='input',kernel_regularizer = regularizers.l1(0.01),bias_regularizer = regularizers.l1(0.01))
    self.list_dense = [self.input_layer]
    self.output_layer = MyDenseLayer(n_outputs,shape = (None,self.n_hiddens),layer_name = 'output')
  def call(self, inputs):
    if self.num_layers>=2:
      for index,layer in enumerate(self.list_dense):
        if index == 0:
          out = layer(inputs,training=False)
          out = tf.nn.relu(out)
        elif index != self.num_layers-1:
          out = layer(out,training=False)
          out = tf.nn.relu(out)
        else:
          out = layer(prev_out+out,training=True)
          out = tf.nn.relu(out)
        prev_out = out
    else:
      out = self.list_dense[0](inputs,training=True)
      out = tf.nn.relu(out)
    out = self.output_layer(out)
    out_sm = tf.nn.softmax(out)
    return out_sm
  def add_layer(self):
    self.num_layers += 1
    new_dense = MyDenseLayer(self.n_hiddens,shape = (None,self.n_hiddens),layer_name =str(self.num_layers),kernel_regularizer = regularizers.l1(0.01),bias_regularizer = regularizers.l1(0.01))
    self.list_dense.append(new_dense)
    for index in range(len(self.layers)-2):
      self.layers[index].trainable = False
  def sparsify_weights(self, threshold = 1e-4):
    weights = self.get_weights()
    sparsified_weights = []
    for index,w in enumerate(weights):
        if index!= len(weights)-4 or index!= len(weights)-3:
          sparsified_weights.append(w)
        else:
          bool_mask = (w > threshold).astype(int)
          sparsified_weights.append(w*bool_mask)
    self.set_weights(sparsified_weights)
