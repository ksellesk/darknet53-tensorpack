# -*- coding: utf-8 -*-
# File: darknet_model.py

import tensorflow as tf
from tensorpack.tfutils.tower import get_current_tower_context

from quant import *
from imagenet_utils import ImageNetModel


leaky_relu = tf.nn.leaky_relu


def fused_conv2d(inputs, filters, kernel_size, strides=(1, 1), activation=leaky_relu, use_bias=False, padding="same"):
    conv2d_out = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)
    batchnorm_out = tf.layers.batch_normalization(conv2d_out, momentum=0.9, epsilon=1e-5, training=get_current_tower_context().is_training)
    actn_out = activation(batchnorm_out)
    return actn_out


def darknet_basicblock(inputs, ch_out, activation=leaky_relu):
    conv1_out = fused_conv2d(inputs, ch_out // 2, 1, activation=activation)
    conv2_out = fused_conv2d(conv1_out, ch_out, 3, activation=activation)
    return inputs + conv2_out


def resnet_group(inputs, num_features, count, activation=leaky_relu):
    for i in range(0, count):
        inputs = darknet_basicblock(inputs, num_features, activation=activation)
    return inputs


def darknet(image, use_fp16, bit_actn, bit_weight):
    if bit_weight is None:
        custom_getter = None
    else:
        custom_getter = quantize_getter_fn(conv_qbit=bit_weight, except_layers=["conv2d/kernel",])
        if bit_actn and bit_actn < 32 and bit_actn > 0:
            actn_fn = lambda i: bit_act(i, bit_actn)
        else:
            actn_fn = leaky_relu

    with tf.variable_scope(tf.get_variable_scope(), custom_getter=custom_getter):
        outs = fused_conv2d(image, 32, 3, strides=1)
        outs = fused_conv2d(outs, 64, 3, strides=2, activation=actn_fn)
        outs = resnet_group(outs, 64, 1, activation=actn_fn)
        outs = fused_conv2d(outs, 128, 3, strides=2, activation=actn_fn)
        outs = resnet_group(outs, 128, 2, activation=actn_fn)
        outs = fused_conv2d(outs, 256, 3, strides=2, activation=actn_fn)
        outs = resnet_group(outs, 256, 8, activation=actn_fn)
        outs = fused_conv2d(outs, 512, 3, strides=2, activation=actn_fn)
        outs = resnet_group(outs, 512, 8, activation=actn_fn)
        outs = fused_conv2d(outs, 1024, 3, strides=2, activation=actn_fn)
        outs = resnet_group(outs, 1024, 4, activation=actn_fn)
        outs = tf.reduce_mean(outs, (1, 2))
        logits = tf.layers.dense(outs, 1000)
    return logits


class Model(ImageNetModel):
    def __init__(self, use_fp16=False, bit_actn=None, bit_weight=None):
        self.use_fp16 = use_fp16
        self.bit_actn = bit_actn
        self.bit_weight = bit_weight

    def get_logits(self, image):
        return darknet(image, self.use_fp16, self.bit_actn, self.bit_weight)
