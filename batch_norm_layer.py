import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class BatchNormLayer:
  def __init__(self, D):
    self.running_mean = tf.Variable(np.zeros(D, dtype=np.float32), trainable=False)
    self.running_var = tf.Variable(np.ones(D, dtype=np.float32), trainable=False)
    self.gamma = tf.Variable(np.ones(D, dtype=np.float32))
    self.beta = tf.Variable(np.zeros(D, dtype=np.float32))

  def forward(self, X):
    return tf.nn.batch_normalization(
      X,
      self.running_mean,
      self.running_var,
      self.beta,
      self.gamma,
      1e-3
    )