"""
HanNet
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import read_data

NUM_FONTS = 8

def deepnn(x):

  with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)
    
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to font classes, one for each digit
  W_fc2 = weight_variable([1024, NUM_FONTS])
  b_fc2 = bias_variable([NUM_FONTS])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

  model_dir = "../data/HanNet_CNN"
  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)
  tf.gfile.MakeDirs(model_dir)
    
  # Load training, validatoin, and eval data
  #[train_set, test_set] = read_data.read_data_sets(False)
  [train_set, validation_set, test_set] = read_data.read_data_sets(True)

  # Build the graph for the deep net
  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, NUM_FONTS])
  y_conv, keep_prob = deepnn(x)
  
  # Define loss and optimizer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
  tf.summary.scalar('cross_entropy', cross_entropy)
  
  #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  
  sess = tf.InteractiveSession()

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(model_dir + '/test')
  
  tf.global_variables_initializer().run()
  for i in range(2000):
    batch = train_set.next_batch(100)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      validation_accuracy = accuracy.eval(feed_dict={x: validation_set.images, y: validation_set.labels, keep_prob: 1.0})
      print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy, validation_accuracy))
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.6})
    train_writer.add_summary(summary, i)

  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_set.images, y: test_set.labels, keep_prob: 1.0}))
  
  model_file = model_dir + "/meta_data/HanNet.meta"
  meta_graph_def = tf.train.export_meta_graph(filename=model_file)
  
if __name__ == '__main__':
  tf.app.run()