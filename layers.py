import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import os
from enum import Enum


# Convolutional layers
def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation

def conv_bn_op(input_op, name, kh, kw, n_out, dh, dw, is_training):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        bn = batch_norm(z, n_out, is_training)
        activation = tf.nn.relu(bn, name=scope)
        return activation


# Batch Normalization
def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


# Upsampling layers
def upsample_op(input_op, name, n_channels, upscale_factor):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.name_scope(name) as scope:
        # Shape of the input bottom tensor
        in_shape = tf.shape(input_op)
        h = ((in_shape[1] - 1) * stride) + 2
        w = ((in_shape[2] - 1) * stride) + 2
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)
        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        weights = get_bilinear_filter(scope, filter_shape, upscale_factor)
        activation = tf.nn.conv2d_transpose(input_op, weights, output_shape, strides=strides, padding='SAME', name=scope)
        return activation


# FC layers
def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation


# Pooling layers
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


# Upscales the weight values.
def get_bilinear_filter(name, filter_shape, upscale_factor):
    # filter_shape is [height, width, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            # Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
            1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    with tf.name_scope(name) as scope:
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        bilinear_weights = tf.get_variable(name=scope + "decon_bilinear_filter",
                                           initializer=init,
                                           shape=weights.shape)
    return bilinear_weights