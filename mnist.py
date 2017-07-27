#!/usr/bin/python
# simple mnist experiment

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import variational_dropout as vd

def deepnn(x, phase):
    """
    Builds the network graph.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
        x: True is train, False is test

    Returns:
        A tuple (y, log_alphas). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). log_alphas is a list of the log_alpha parameters describing
        the effective dropout rate of the approximate posterior.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.variable_scope('conv1'):
        h_conv1 = vd.conv2d(x_image, 32, [5,5], phase)
    
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.variable_scope('conv2'):
        h_conv2 = vd.conv2d(h_pool1, 64, [5,5], phase)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.variable_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = vd.fully_connected(h_pool2_flat, 1024, phase)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.variable_scope('fc2'):
        y_conv = vd.fully_connected(h_fc1, 10, phase) 
    return y_conv

def main():
    # Import data
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

    # Define placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    phase = tf.placeholder(tf.bool, None)

    # Build the graph for the deep net
    y_conv = deepnn(x, phase)

    with tf.name_scope('loss'):
        # cross entropy part of the ELBO
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv))
        # prior DKL part of the ELBO
        log_alphas = [v for v in tf.global_variables() if 'log_alpha' in v.name]
        import pdb
        pdb.set_trace()
        divergences = [vd.dkl_qp(la) for la in log_alphas]
        # combine to form the ELBO
        N = float(mnist.train.images.shape[0])
        dkl = tf.reduce_sum(tf.stack(divergences))
        elbo = cross_entropy+(1./N)*dkl

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(elbo)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
    with tf.name_scope('sparseness'):
        sparse = sparseness(log_alphas)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 1000 == 0:
                train_accuracy, train_loss = sess.run((accuracy, cross_entropy),
                    feed_dict={x: batch[0], y_: batch[1], phase: False})
                print('step %d, training accuracy %g, training loss: %g' %
                    (i, train_accuracy, train_loss))
                val_x, val_y = mnist.validation.next_batch(50)
                val_accuracy, val_loss, val_sp = sess.run((accuracy, cross_entropy, sparse),
                    feed_dict={x: val_x, y_: val_y, phase: False})
                print('step %d, val accuracy %g, val loss: %g, sparsity: %g' %
                    (i, val_accuracy, val_loss, val_sp))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], phase: True})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, phase: False}))

if __name__ == '__main__':
    main()
