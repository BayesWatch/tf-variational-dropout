#!/usr/bin/python

import tensorflow as tf

# required operations
def paranoid_log(x, eps=1e-8):
    return tf.log(x+eps)

def clip(x):
    return tf.clip_by_value(x, -8., 8.)

def get_log_alpha(log_sigma2, w):
    log_alpha = clip(log_sigma2 - paranoid_log(tf.square(w)))
    return tf.identity(log_alpha, name='log_alpha')

#def vardrop_fc(x, log_alpha, w, phase, thresh=3):
def fully_connected(x, phase, n_hidden, activation_fn=tf.nn.relu, thresh=3,
        initializer=tf.contrib.layers.xavier_initializer):
    # you get xavier initialization, and that's it for now
    n_input = int(x.shape[1])
    w = tf.get_variable("w", [n_input, n_hidden],
            initializer=initializer())
    b = tf.get_variable("b", [n_hidden,],
            initializer=tf.constant_initializer(0.))
    log_sigma2 = log_sigma2_variable([n_input, n_hidden])
    log_alpha = get_log_alpha(log_sigma2, w)

    # at test time,we just mask
    select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)

    # choose between adding noise, or applying mask, depending on phase
    activations = tf.cond(phase, lambda: fc_noisy(x, log_alpha, w), lambda: fc_masked(x, select_mask, w))
    return activation_fn(activations + b)

def fc_noisy(x, log_alpha, w):
    mu = tf.matmul(x, w)
    si = tf.sqrt(tf.matmul(tf.square(x), tf.exp(log_alpha)*tf.square(w))+1e-8)
    return mu + si*tf.random_normal(tf.shape(mu))

def fc_masked(x, select_mask, w):
    return tf.matmul(x, w*select_mask)

#def vardrop_conv2d(x, log_alpha, w, phase, thresh=3):
def conv2d(x, phase, n_filters, kernel_size, activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.xavier_initializer_conv2d, thresh=3):
    n_input_channels = int(x.shape[3])
    # define parameters
    conv_param_shape = kernel_size+[n_input_channels, n_filters]
    w = tf.get_variable("w", conv_param_shape,
            initializer=initializer())
    b = tf.get_variable("b", [n_filters],
            initializer=tf.constant_initializer())
    log_sigma2 = log_sigma2_variable(conv_param_shape)
    log_alpha = get_log_alpha(log_sigma2, w)

    select_mask = tf.cast(tf.less(log_alpha, thresh), tf.float32)
    
    activations = tf.cond(phase, lambda: conv2d_noisy(x, log_alpha, w), lambda: conv2d_masked(x, select_mask, w))
    return activation_fn(activations + b)

def conv2d_noisy(x, log_alpha, w):
    conved_mu = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    conved_si = tf.sqrt(tf.nn.conv2d(tf.square(x),
                                     tf.exp(log_alpha)*tf.square(w),
                                     strides=[1, 1, 1, 1], padding='SAME')+1e-8)
    return conved_mu + tf.random_normal(tf.shape(conved_mu))*conved_si

def conv2d_masked(x, select_mask, w):
    return tf.nn.conv2d(x, w*select_mask, strides=[1, 1, 1, 1], padding='SAME')

# homemade initializers
def log_sigma2_variable(shape, ard_init=-10.):
    return tf.get_variable("log_sigma2", shape=shape,
            initializer=tf.constant_initializer(ard_init))

def dkl_qp(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695; C = -k1
    mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(tf.exp(-log_alpha)) + C
    return -tf.reduce_sum(mdkl)

# handy function to keep track of sparsity
def sparseness(log_alphas, thresh=3):
    N_active, N_total = 0., 0.
    for la in log_alphas:
        m = tf.cast(tf.less(la, thresh), tf.float32)
        n_active = tf.reduce_sum(m)
        n_total = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
        N_active += n_active
        N_total += n_total
    return 1.0 - N_active/N_total

# utility to gather variational dropout parameters
def gather_logalphas(graph):
    node_defs = [n for n in graph.as_graph_def().node if 'log_alpha' in n.name]
    tensors = [graph.get_tensor_by_name(n.name+":0") for n in node_defs]
    return tensors
