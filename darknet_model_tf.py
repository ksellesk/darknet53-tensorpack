# -*- coding: utf-8 -*-
# File: darknet_model.py

import tensorflow as tf

from imagenet_utils import ImageNetModel

from tensorpack.tfutils.tower import get_current_tower_context

leaky_relu = tf.nn.leaky_relu


def fused_conv2d(inputs, filters, kernel_size, strides=(1, 1), activation=leaky_relu, use_bias=False, padding="same"):
    conv2d_out = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)
    batchnorm_out = tf.layers.batch_normalization(conv2d_out, momentum=0.9, epsilon=1e-5, training=get_current_tower_context().is_training)
    actn_out = activation(batchnorm_out)
    return actn_out


def darknet_basicblock(inputs, ch_out):
    conv1_out = fused_conv2d(inputs, ch_out // 2, 1)
    conv2_out = fused_conv2d(conv1_out, ch_out, 3)
    return inputs + conv2_out


def resnet_group(inputs, num_features, count):
    for i in range(0, count):
        inputs = darknet_basicblock(inputs, num_features)
    return inputs


def darknet(image, use_fp16):
    outs = fused_conv2d(image, 32, 3, strides=1)
    outs = fused_conv2d(outs, 64, 3, strides=2)
    outs = resnet_group(outs, 64, 1)
    outs = fused_conv2d(outs, 128, 3, strides=2)
    outs = resnet_group(outs, 128, 2)
    outs = fused_conv2d(outs, 256, 3, strides=2)
    outs = resnet_group(outs, 256, 8)
    outs = fused_conv2d(outs, 512, 3, strides=2)
    outs = resnet_group(outs, 512, 8)
    outs = fused_conv2d(outs, 1024, 3, strides=2)
    outs = resnet_group(outs, 1024, 4)
    outs = tf.reduce_mean(outs, (1, 2))
    logits = tf.layers.dense(outs, 1000)
    return logits


class Model(ImageNetModel):
    def __init__(self, use_fp16=False):
        self.use_fp16 = use_fp16

    def get_logits(self, image):
        return darknet(image, self.use_fp16)
