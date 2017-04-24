"""
HanNet
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
import HanNet_input
import HanNet_params as PARAMS

MODEL_DIR   = PARAMS.model_dir
SAVER_NAME  = PARAMS.saver_name
NUM_FONTS   = len(PARAMS.fonts)
PIC_SIZE    = PARAMS.pic_size
BATCH_SIZE  = PARAMS.batch_size
TOTAL_STEPS = PARAMS.total_steps
KEEP_RATIO  = 1.0 - PARAMS.drop_ratio

NUM_THREADS = PARAMS.num_threads
USE_GPU     = PARAMS.use_gpu

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv_layer(name, x, filter_size, in_filters, out_filters):
  with tf.variable_scope(name):
    n = filter_size * filter_size * in_filters
    kernel = tf.get_variable(name='DW',
                             shape=[filter_size, filter_size, in_filters, out_filters],
                             dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
    bias = tf.get_variable(name='bias',
                           shape=[out_filters],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.1))
    conv_map = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    pre_act  = tf.nn.bias_add(conv_map, bias)
    post_act = tf.nn.relu(pre_act)
    return post_act
    

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def fully_connected(name, x, fan_in, fan_out, useRelu=True):
  with tf.variable_scope(name):
    weights = tf.get_variable(name='DW',
                              shape=[fan_in, fan_out],
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/fan_in)))
    bias = tf.get_variable(name='bias',
                           shape=[fan_out],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.1))
    if x.get_shape()[1] != fan_in:
      x = tf.reshape(x, [-1, fan_in])
    pre_act = tf.nn.xw_plus_b(x, weights, bias)
    post_act = tf.nn.relu(pre_act)
    if useRelu:
      return post_act
    else:
      return pre_act
    

def toy_cnn(x, keep_prob):

  x_image = tf.reshape(x, [-1, PIC_SIZE, PIC_SIZE, 1])
  tf.summary.image('input', x_image, min(25, x.shape[0]))
    
  net = conv_layer('conv1', x_image, 5, 1, 32) 

  net = max_pool_2x2(net)

  net = conv_layer('conv2', net, 5, 32, 64) 

  net = max_pool_2x2(net)

  net = fully_connected('fc1', net, (PIC_SIZE>>2)*(PIC_SIZE>>2)*64, 1024)

  net = tf.nn.dropout(net, keep_prob)

  logits = fully_connected('final', net, 1024, NUM_FONTS, useRelu=False)

  return logits


def vgg_lite(x, keep_prob):

  x_image = tf.reshape(x, [-1, PIC_SIZE, PIC_SIZE, 1])
  tf.summary.image('input', x_image, min(25, x.shape[0]))
    
  net = conv_layer('conv1a', x_image, 3, 1, 32) 
  net = conv_layer('conv1b', net, 3, 32, 32) 

  net = max_pool_2x2(net)

  net = conv_layer('conv2a', net, 3, 32, 64) 
  net = conv_layer('conv2b', net, 3, 64, 64) 
  net = conv_layer('conv2c', net, 3, 64, 64) 

  net = max_pool_2x2(net)

  net = conv_layer('conv3a', net, 3, 64, 128) 
  net = conv_layer('conv3b', net, 3, 128, 128) 
  net = conv_layer('conv3c', net, 3, 128, 128) 

  net = max_pool_2x2(net)

  net = fully_connected('fc1', net, (PIC_SIZE>>3)*(PIC_SIZE>>3)*128, 1024)

  net = tf.nn.dropout(net, keep_prob)

  net = fully_connected('fc2', net, 1024, 1024)

  net = tf.nn.dropout(net, keep_prob)

  logits = fully_connected('final', net, 1024, NUM_FONTS, useRelu=False)

  return logits


def main(_):
  if tf.gfile.Exists(MODEL_DIR):
    tf.gfile.DeleteRecursively(MODEL_DIR)
  tf.gfile.MakeDirs(MODEL_DIR)

  if USE_GPU is True:
    dev = '/gpu:0'
  else:
    dev = '/cpu:0'

  # Load training, validatoin, and eval data
  #[train_set, test_set] = HanNet_input.read_data_sets(False)
  [train_set, validation_set, test_set] = HanNet_input.read_data_sets(True)

  # Build the graph for the deep net
  x = tf.placeholder(tf.float32, [None, PIC_SIZE*PIC_SIZE], name='x-input')
  y = tf.placeholder(tf.float32, [None, NUM_FONTS], name='y-input')
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  with tf.device(dev):
    logits = toy_cnn(x, keep_prob)
    #logits = vgg_lite(x, keep_prob)
    
  # Define loss and optimizer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy)
  
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  #train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
  tf.summary.scalar('accuracy', accuracy)
  
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(MODEL_DIR + '/train')
  test_writer = tf.summary.FileWriter(MODEL_DIR + '/test')
  
  saver = tf.train.Saver()
  sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                                                     allow_soft_placement=True))

  tf.global_variables_initializer().run()
  for i in range(TOTAL_STEPS):
    batch = train_set.next_batch(BATCH_SIZE)
    if not i % 50:
      train_loss, train_accuracy = sess.run([cross_entropy, accuracy], 
                                            feed_dict={'x-input:0': batch[0],
                                                       'y-input:0': batch[1],
                                                       'keep_prob:0': 1.0})
      validation_accuracy = accuracy.eval(feed_dict={'x-input:0': validation_set.images,
                                                     'y-input:0': validation_set.labels,
                                                     'keep_prob:0': 1.0})
      print('step %d, training accuracy %g, validation accuracy %g' %
            (i, train_accuracy, validation_accuracy))
      #print('step %d: training loss %g, accuracy %g' % (i, train_loss, train_accuracy))

    if not i % 10:
      summary = sess.run(merged, feed_dict={'x-input:0': batch[0],
                                            'y-input:0': batch[1],
                                            'keep_prob:0': 1.0})
      train_writer.add_summary(summary, i)

    sess.run(train_step, feed_dict={'x-input:0': batch[0],
                                    'y-input:0': batch[1],
                                    'keep_prob:0': KEEP_RATIO})

  print('test accuracy %g' %
        accuracy.eval(feed_dict={'x-input:0': test_set.images,
                                 'y-input:0': test_set.labels,
                                 'keep_prob:0': 1.0}))
  
  saver.save(sess, SAVER_NAME)
  
if __name__ == '__main__':
  tf.app.run()
