# custom initializer in the style of pytorch's

import math

# call it by the original class name
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.python.framework import dtypes

def pytorch_initializer(uniform=True, seed=None, dtype=dtypes.float32):
  return variance_scaling_initializer(factor=1./3, mode='FAN_IN', uniform=uniform, seed=seed, dtype=dtype)
