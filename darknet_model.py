# -*- coding: utf-8 -*-
# File: darknet_model.py

import tensorflow as tf

from tensorpack.common import layer_register
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, FullyConnected)

from imagenet_utils import ImageNetModel


@layer_register(use_scope=None)
def BNLeakyReLU(x, name=None):
    """
    A shorthand of BatchNormalization + LeakyReLU.
    """
    x = BatchNorm('bn', x)
    x = tf.nn.leaky_relu(x, name=name)
    return x


def darknet_basicblock(l, ch_out):
    shortcut = l
    l = Conv2D('conv0', l, ch_out // 2, 1, activation=BNLeakyReLU)
    l = Conv2D('conv1', l, ch_out, 3, activation=BNLeakyReLU)
    return l + shortcut


def resnet_group(name, l, num_features, count):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = darknet_basicblock(l, num_features)
    return l


def darknet(image, use_fp16):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        l = Conv2D('conv0', image, 32, 3, strides=1, activation=BNLeakyReLU)
        l = Conv2D('conv1', l, 64, 3, strides=2, activation=BNLeakyReLU)
        l = resnet_group('group0', l, 64, 1)
        l = Conv2D('conv2', l, 128, 3, strides=2, activation=BNLeakyReLU)
        l = resnet_group('group1', l, 128, 2)
        l = Conv2D('conv3', l, 256, 3, strides=2, activation=BNLeakyReLU)
        l = resnet_group('group2', l, 256, 8)
        l = Conv2D('conv4', l, 512, 3, strides=2, activation=BNLeakyReLU)
        l = resnet_group('group3', l, 512, 8)
        l = Conv2D('conv5', l, 1024, 3, strides=2, activation=BNLeakyReLU)
        l = resnet_group('group4', l, 1024, 4)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits


class Model(ImageNetModel):
    def __init__(self, use_fp16=False):
        self.use_fp16 = use_fp16

    def get_logits(self, image):
        return darknet(image, self.use_fp16)
