import tensorflow as tf
import numpy as np
import os
from util import data_reader, loss
from models.processing_tools import compute_accuracy

class base(object):
    def __init__(self, args):
        self.channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        #Model name
        self.model_name = 'base'

        #Model parms
        self.rnn_cells = 20
        self.batch_size = 10
        self.input_H = 512
        self.input_W = 512
        self.featmap_H = 16
        self.featmap_W = 16

        self.num_vocab = 8803
        self.embed_dim = 1000
        self.lstm_dim = 1000
        self.mlp_hidden_dims = 500

        #Training parms
        self.max_iter = 1
        self.start_lr = 0.01
        self.lr_decay_step = 6000
        self.lr_decay_rate = 0.1
        self.weight_decay = 0.0005
        self.momentum = 0.9

        #which dataset to use, e.g. referit, coco, coco+, cocoref
        self.dataset = 'referit'

        self.data_folder = './data/' + self.dataset + '/train_batch_seg'
        self.data_prefix = "referit_train_seg"

        self.log_folder = './log/' + self.dataset

        self.snapshot = 5000


        #args for subclass models
        self.args = args

    def forward(self, imcrop_batch, text_seq_batch, is_training=True, model='base'):
        """
        This function forward the inputs and return the tensor of mask score
        Subclass model must override this method.
        """
        raise NotImplementedError('Model must implment forward method')

    def get_train_var_list(self):
        """
        This function returns the list of variables to train. Subclass should modify accordingly.
        """
        return [var for var in tf.trainable_variables()]

    def build_model(self):
        self.imcrop_batch = tf.placeholder(tf.float32, [self.batch_size, self.input_H, self.input_W, 3])
        self.text_seq_batch = tf.placeholder(tf.int32, [self.rnn_cells, self.batch_size])
        self.label_batch = tf.placeholder(tf.float32, [self.batch_size, self.input_H, self.input_W, 1])
        self.scores = self.forward(self.imcrop_batch, self.text_seq_batch)

        self.train_var_list = self.get_train_var_list()

        #Calculate loss
        self.cls_loss = loss.weighed_logistic_loss(self.scores, self.label_batch)

        # Add regularization to weight matrices (excluding bias)
        reg_var_list = [var for var in tf.trainable_variables()
                if (var in self.train_var_list) and
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]

        reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.total_loss = self.cls_loss + reg_loss

    def train_op(self, loss, train_var_list):
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.start_lr, global_step, self.lr_decay_step, 
            self.lr_decay_rate, staircase=True)
        solver = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
        grads_and_vars = solver.compute_gradients(loss, var_list=self.train_var_list)
        train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)
        return train_step

    def initialize(self):
        """
        This function run customized initialize operations that subclass override.
        """
        pass

    def train(self):
        #Build model, and get train_op
        self.build_model()
        train_op = self.train_op(self.total_loss, self.get_train_var_list())

        reader = data_reader.DataReader(self.data_folder, self.data_prefix)

        cls_loss_avg = 0
        avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
        decay = 0.99

        #Add summary for tensorboard




        # tf.train.Saver is used to save and load intermediate models.
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)

        sess = tf.Session()
        self.sess = sess

        #Run initialization operations
        sess.run(tf.global_variables_initializer())
        self.initialize()

        for n_iter in range(self.max_iter):
            batch = reader.read_batch()
            text_seq_val = batch['text_seq_batch']
            imcrop_val = batch['imcrop_batch'].astype(np.float32) - self.channel_mean
            label_val = batch['label_fine_batch'].astype(np.float32)

            # Forward and Backward pass
            scores_val, cls_loss_val, _, lr_val = sess.run([self.scores, self.cls_loss, train_op, self.learning_rate],
                feed_dict={
                    self.text_seq_batch  : text_seq_val,
                    self.imcrop_batch    : imcrop_val,
                    self.label_batch     : label_val
                })
            cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
            print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, lr = %f'
                % (n_iter, cls_loss_val, cls_loss_avg, lr_val))

            # Accuracy
            accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
            avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
            avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
            avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
            print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
                % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
            print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

            if (n_iter + 1) % self.snapshot == 0 or (n_iter+1) >= self.max_iter:
                checkpoint_path = os.path.join(self.log_folder, 'checkpoints')
                self.save(checkpoint_path, n_iter)

    def load(self, checkpoint_dir='checkpoints', step=None):
        pass

    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def test(self, data_list):
        pass
