# custom initializer in the style of pytorch's

import math

# call it by the original class name
import tensorflow as tf

class PyTorchInit(tf.random_uniform_initializer):
    def __call__(self, shape, dtype=None, partition_info=None):
        n = shape[2] # in_channels
        for k in shape[:2]:
            n *= k # kernel size
        stdv = 1./math.sqrt(n)
        self.minval = -stdv
        self.maxval = stdv
        super(PyTorchInit, self).__call__(shape, dtype=dtype, partition_info=partition_info)
