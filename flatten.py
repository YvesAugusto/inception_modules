import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class Flatten:
  def forward(self, X):
    return tf.contrib.layers.flatten(X)

  def get_params(self):
    return []