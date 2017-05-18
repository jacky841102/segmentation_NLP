import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
def deeplab_pool5(input_batch, name):
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1):
            conv1_1 = slim.conv2d(input_batch, 64, [3, 3], scope='conv1_1')
            conv1_2 = slim.conv2d(conv1_1, 64, [3, 3], scope='conv1_2')
            pool1 = slim.max_pool2d(conv1_2, [3, 3], stride=2, padding='SAME', scope='pool1')

            conv2_1 = slim.conv2d(pool1, 128, [3, 3], scope='conv2_1')
            conv2_2 = slim.conv2d(conv2_1, 128, [3, 3], scope='conv2_2')
            pool2 = slim.max_pool2d(conv2_2, [3, 3], stride=2, padding='SAME', scope='pool2')
        
            conv3_1 = slim.conv2d(pool2, 256, [3, 3], scope='conv3_1')
            conv3_2 = slim.conv2d(conv3_1, 256, [3, 3], scope='conv3_2')
            conv3_3 = slim.conv2d(conv3_2, 256, [3, 3], scope='conv3_3')
            pool3 = slim.max_pool2d(conv3_3, [3, 3], stride=2, padding='SAME', scope='pool3')
            
            conv4_1 = slim.conv2d(pool3, 512, [3, 3], scope='conv4_1')
            conv4_2 = slim.conv2d(conv4_1, 512, [3, 3], scope='conv4_2')
            conv4_3 = slim.conv2d(conv4_2, 512, [3, 3], scope='conv4_3')
            pool4 = slim.max_pool2d(conv4_3, [3, 3], stride=1, padding='SAME', scope='pool4')

            conv5_1 = slim.conv2d(pool4, 512, [3, 3], scope='conv5_1')
            conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], scope='conv5_2')
            conv5_3 = slim.conv2d(conv5_2, 512, [3, 3], scope='conv5_3')
            pool5 = slim.max_pool2d(conv5_3, [3, 3], stride=1, padding='SAME', scope='pool5')
            pool5a = slim.avg_pool2d(pool5, [3, 3], stride=1, padding='SAME', scope='pool5a')

            return pool5a


def deeplab_fc8(input_batch, name, apply_dropout=False):
    pool5a = deeplab_pool5(input_batch, name)
    with tf.variable_scope(name):
        fc6 = fc_relu('fc6', pool5a, output_dim=1024)
        if apply_dropout: fc6 = drop(fc6, 0.5)

        fc7 = fc_relu('fc7', fc6, output_dim=1024)
        if apply_dropout: fc7 = drop(fc7, 0.5)
        fc8 = fc('fc8', fc7, output_dim=1000)
        return fc8

def deeplab_fc8_full_conv(input_batch, name, apply_dropout=False, output_dim=1000):
    pool5a = deeplab_pool5(input_batch, name)
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1):
            fc6 = slim.conv2d(pool5a, 1024, [3, 3], rate=12, padding='SAME', scope='fc6')
            if apply_dropout: fc6 = tf.nn.dropout(fc6, 0.5)

            fc7 = slim.conv2d(fc6, 1024, [1, 1], scope='fc7')
            if apply_dropout: fc7 = tf.nn.dropout(fc7, 0.5)

        fc8 = slim.conv2d(fc7, output_dim, [1, 1], activation_fn=None, scope='fc8_voc12')
        return fc8

