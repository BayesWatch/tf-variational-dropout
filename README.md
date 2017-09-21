
Tensorflow (incomplete) replication of the paper [Variational Dropout
Sparsifies Deep Neural Networks][paper], based on [their code][code].
Written to be relatively easy to apply.

How to Use
==========


This only implements two types of layers at the moment, fully connected and
2D convolutional. Example usage is in `mnist.py`. We are following the
[tensorflow docs on variable reuse][docs], so individual layers must have
their own `variable_scope`. So, from the `mnist.py` script:

```
import variational_dropout as vd

with tf.variable_scope('fc2'):
    y_conv = vd.fully_connected(h_fc1, phase, 10) 
```

The `phase` variable is used to switch between training and test time
behaviours, typically using a placeholder. `True` is training time, and the
noise variables will be sampled based on the current variational
parameters. `False` is test time, and weights will be masked based on the
current variational parameters.  Training time is stochastic, while test is
deterministic.

To train with variational dropout, the loss function must also include the
KL divergence between the approximate posterior and the prior. You can
think of this as a (kind of) theoretically justified regulariser. There is
a function to gather the `log_alpha` (`vd.gather_log_alphas()`) variables
that parameterise the approximate posterior and another to estimate this KL
divergence. A typical way to calculate it and add it to the loss is given
in the `mnist.py` script:

```
# prior DKL part of the ELBO
log_alphas = vd.gather_logalphas(tf.get_default_graph())
divergences = [vd.dkl_qp(la) for la in log_alphas]
# combine to form the ELBO
N = float(mnist.train.images.shape[0])
dkl = tf.reduce_sum(tf.stack(divergences))
elbo = cross_entropy+(1./N)*dkl
```

This is *not scaled correctly* to be a true ELBO, but it's not really
relevant considering the arbitrary choice of learning rate.

[paper]: https://arxiv.org/abs/1701.05369
[code]: https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn
[docs]: https://www.tensorflow.org/programmers_guide/variable_scope
