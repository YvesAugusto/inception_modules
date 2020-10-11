import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class AvgPool:
  def __init__(self, ksize):
    self.ksize = ksize

  def forward(self, X):
    return tf.nn.avg_pool(
      X,
      ksize=[1, self.ksize, self.ksize, 1],
      strides=[1, 1, 1, 1],
      padding='VALID'
    )

  def get_params(self):
    return []