import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from util import data_reader, loss
from models.processing_tools import compute_accuracy

class base(object):
    """
    This class is the base class of all experiment models.
    Subclass must implement forward().
    Subclass can override initialize() and get_train_var_list() method.
    """

    def __init__(self, **kwargs):
        """
        Init metod.
        :kwargs is a dict containing all arguments for the model
        """       
        self.channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

        #Model name
        self.model_name = 'base'

        #Model parms
        self.rnn_cells = kwargs['rnn_cells'] if 'rnn_cells' in kwargs else 20
        self.input_H = kwargs['input_H'] if 'input_H' in kwargs else 512
        self.input_W = kwargs['input_W'] if 'input_W' in kwargs else 512
        self.featmap_H = kwargs['featmap_H'] if 'featmap_H' in kwargs else 16
        self.featmap_W = kwargs['featmap_W'] if 'featmap_W' in kwargs else 16

        self.num_vocab = kwargs['num_vocab'] if 'num_vocab' in kwargs else 8803
        self.embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 1000
        self.lstm_dim = kwargs['lstm_dim'] if 'lstm_dim' in kwargs else 1000
        self.mlp_hidden_dims = kwargs['mlp_hidden_dims'] if 'mlp_hidden_dims' in kwargs else 500

        #Training parms
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10
        self.max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 30000
        self.start_lr = kwargs['start_lr'] if 'start_lr' in kwargs else 0.01
        self.lr_decay_step = kwargs['lr_decay_step'] if 'lr_decay_step' in kwargs else 6000
        self.lr_decay_rate = kwargs['lr_decay_rate'] if 'lr_decay_rate' in kwargs else 0.1
        self.weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0.0005
        self.momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9

        #which dataset to use, e.g. referit, coco, coco+, cocoref
        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'referit'

        self.data_folder = './data/%s/train_batch_seg' % self.dataset
        self.data_prefix = "%s_train_seg" % self.dataset

        self.log_folder = './log/%s' % self.dataset
        self.log_step = kwargs['log_step'] if 'log_step' in kwargs else 10
        self.checkpoint_step = kwargs['checkpoint_step'] if 'checkpoint_step' in kwargs else 5000

        #args for subclass models
        self.kwargs = kwargs

    def forward(self, imcrop_batch, text_seq_batch, is_training=True):
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

        #Add regularization to weight matrices (excluding bias)
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

    def initialize(self, sess):
        """
        This function run customized initialization operations that subclass override.
        """
        pass

    def log_info(self):
        print('################################################')
        print(time.ctime())
        print('Training model: %s' % self.model_name)

        print('Model parms:')
        print('\tinput_H: %d' % self.input_H)
        print('\tinput_W: %d' % self.input_W)
        print('\tfeatmap_H: %d' % self.featmap_H)
        print('\tfeatmap_W: %d' % self.featmap_W)
        print('\trnn_cells: %d' % self.rnn_cells)

        print('Training parms:')
        print('\tbatch_size: %d' % self.batch_size)
        print('\tmax_iter: %d' % self.max_iter)
        print('\tstart_lr: %f' % self.start_lr)
        print('\tlr_decay_step: %f' % self.lr_decay_step)
        print('\tlr_decay_rate: %f' % self.lr_decay_rate)
        print('\tweight_decay: %f' % self.weight_decay)
        print('\tmomentum: %f' % self.momentum)

        print('Variables for training:')
        for var in self.train_var_list:
            print('\t%s' % var.name)
        print('################################################')

    def train(self):
        #Build model, and get train_op
        self.build_model()
        train_op = self.train_op(self.total_loss, self.get_train_var_list())

        reader = data_reader.DataReader(self.data_folder, self.data_prefix)

        cls_loss_avg = 0
        avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
        decay = 0.99

        #Accuracy palceholder
        acc_all, acc_pos, acc_neg = tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())
        acc_all_avg, acc_pos_avg, acc_neg_avg = tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())
        

        #Add summary for tensorboard
        tf.summary.scalar('loss', self.cls_loss, ['train'])
        tf.summary.scalar('learning_rate', self.learning_rate, ['train'])
        tf.summary.scalar('accuracy_all', acc_all, ['acc'])
        tf.summary.scalar('accuracy_positive', acc_pos, ['acc'])
        tf.summary.scalar('accuracy_negative', acc_neg, ['acc'])
        tf.summary.scalar('accuracy_all_average', acc_all_avg, ['acc'])
        tf.summary.scalar('accuracy_positive_average', acc_pos_avg, ['acc'])
        tf.summary.scalar('accuracy_negative_average', acc_neg_avg, ['acc'])
        train_summary = tf.summary.merge_all(key='train')
        acc_summary = tf.summary.merge_all(key='acc')

        # tf.train.Saver is used to save and load intermediate models.
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)

        sess = tf.Session()
        self.sess = sess

        # Init train_writer
        train_writer = tf.summary.FileWriter(self.log_folder, sess.graph)

        #Run initialization operations
        sess.run(tf.global_variables_initializer())
        self.initialize(sess)

        for n_iter in range(self.max_iter):
            batch = reader.read_batch()
            text_seq_val = batch['text_seq_batch']
            imcrop_val = batch['imcrop_batch'].astype(np.float32) - self.channel_mean
            label_val = batch['label_fine_batch'].astype(np.float32)

            start_time = time.time()

            # Forward and Backward pass
            scores_val, cls_loss_val, _, lr_val, train_sum = sess.run([self.scores, self.cls_loss, train_op, self.learning_rate, train_summary],
                feed_dict={
                    self.text_seq_batch  : text_seq_val,
                    self.imcrop_batch    : imcrop_val,
                    self.label_batch     : label_val
                })
            
            duration = time.time() - start_time

            cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val

            # Accuracy
            accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
            avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
            avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
            avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg

            if (n_iter + 1) % self.log_step == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: iter %d, %.1f data/sec; %.3f '
                              'sec/batch')
                print (format_str % (datetime.now(), n_iter,
                                     examples_per_sec, sec_per_batch))
                                     
                print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, lr = %f'
                    % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
                print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
                    % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
                print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                    % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

                #Fillin placeholder of accuracy
                acc_sum = sess.run(acc_summary, 
                    feed_dict={
                        acc_all: accuracy_all,
                        acc_pos: accuracy_pos,
                        acc_neg: accuracy_neg,
                        acc_all_avg: avg_accuracy_all,
                        acc_pos_avg: avg_accuracy_pos,
                        acc_neg_avg: avg_accuracy_neg
                    }
                )

                train_writer.add_summary(train_sum, n_iter)
                train_writer.add_summary(acc_sum, n_iter)

            if (n_iter + 1) % self.checkpoint_step == 0 or (n_iter+1) >= self.max_iter:
                checkpoint_path = os.path.join(self.log_folder, 'checkpoints')
                self.save(checkpoint_path, n_iter + 1)

    def load(self, checkpoint_dir='checkpoints', step=None):
        pass

    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def test(self, data_list):
        pass
