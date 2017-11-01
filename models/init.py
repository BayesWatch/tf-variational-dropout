# custom initializer in the style of pytorch's

import math

# call it by the original class name
from tensorflow.initializers import random_uniform as RandomUniform
from tensorflow.python.ops import random_ops

class PyTorchInit(RandomUniform):
    def __call__(self, shape, dtype=None, partition_info=None):
        n = shape[2] # in_channels
        for k in shape[:2]:
            n *= k # kernel size
        stdv = 1./math.sqrt(n)
        if dtype is None:
            dtype = self.dtype
        return random_ops.random_uniform(
            shape, -stdv, stdv, dtype, seed=self.seed)
