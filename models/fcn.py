import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models.base import *
from models.components import vgg_net, lstm_net
from models.processing_tools import *

class FCN(base):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.model_name = 'fcn'

    def forward(self, imcrop_batch, text_seq_batch, is_training=True):
        num_vocab, embed_dim, lstm_dim, mlp_hidden_dims = self.num_vocab, self.embed_dim, self.lstm_dim, self.mlp_hidden_dims
        vgg_dropout = self.kwargs['vgg_dropout'] if 'vgg_dropout' in self.kwargs else False
        mlp_dropout = self.kwargs['mlp_dropout'] if 'mlp_dropout' in self.kwargs else False

        with tf.variable_scope(self.model_name):
            # Language feature (LSTM hidden state)
            feat_lang = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)[0]

            # Local image feature
            feat_vis = vgg_net.vgg_fc8_full_conv(imcrop_batch, 'vgg_local',
                apply_dropout=vgg_dropout)

            # Reshape and tile LSTM top
            featmap_H, featmap_W = feat_vis.get_shape().as_list()[1:3]
            N, D_text = feat_lang.get_shape().as_list()
            feat_lang = tf.tile(tf.reshape(feat_lang, [N, 1, 1, D_text]),
                [1, featmap_H, featmap_W, 1])

            # L2-normalize the features (except for spatial_batch)
            # and concatenate them along axis 3 (channel dimension)
            spatial_batch = tf.convert_to_tensor(generate_spatial_batch(N, featmap_H, featmap_W))
            feat_all = tf.concat(axis=3, values=[tf.nn.l2_normalize(feat_lang, 3),
                                tf.nn.l2_normalize(feat_vis, 3),
                                spatial_batch])

            # MLP Classifier over concatenate feature
            with tf.variable_scope('classifier'):
                mlp_l1 = conv_relu('mlp_l1', feat_all, kernel_size=1, stride=1,
                    output_dim=mlp_hidden_dims)
                if mlp_dropout:
                    mlp_l1 = drop(mlp_l1, 0.5)
                mlp_l2 = conv('mlp_l2', mlp_l1, kernel_size=1, stride=1, output_dim=1) 

                upsample8s = deconv('upsample8s', mlp_l2, kernel_size=16,
                    stride=8, output_dim=1, bias_term=False)
                upsample32s = deconv('upsample32s', upsample8s, kernel_size=8,
                    stride=4, output_dim=1, bias_term=False)

        return upsample32s

    def get_train_var_list(self):
        fix_convnet = self.kwargs['fix_convnet'] if 'fix_convnet' in self.kwargs else True

        if fix_convnet:
            return [var for var in tf.trainable_variables() if 'fcn/vgg_local/conv' not in var.name]
        else:
            return [var for var in tf.trainable_variables()]
        

    def initialize(self, sess):
        pretrained_file = self.kwargs['pretrained_file'] if 'pretrained_file' in self.kwargs \
                                else 'models/components/pretrained/vgg_params.npz'

        convnet_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3',
                  'conv4_1', 'conv4_2', 'conv4_3',
                  'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7']

        pretrained_parms = np.load(pretrained_file)

        pretrained_W = pretrained_parms['processed_W'][()]
        pretrained_B = pretrained_parms['processed_B'][()]

        assign_ops = []
        with tf.variable_scope('fcn/vgg_local', reuse=True):
            for l_name in convnet_layers:
                w = tf.get_variable(l_name + '/weights')
                assign_W = tf.assign(w, pretrained_W[l_name].reshape(w.get_shape().as_list()))
                assign_B = tf.assign(tf.get_variable(l_name + '/biases'), pretrained_B[l_name])
                assign_ops += [assign_W, assign_B]

        sess.run(tf.group(*assign_ops))

            
