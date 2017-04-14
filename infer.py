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

def main(_):

  model_dir = "../data/HanNet_CNN"
  saver_name = model_dir + "/HanNet"
  meta_file = saver_name + ".meta"
    
  # Load training, validatoin, and eval data
  #[train_set, test_set] = read_data.read_data_sets(False)
  [train_set, validation_set, test_set] = read_data.read_data_sets(True)

  saver = tf.train.import_meta_graph(meta_file)
  sess = tf.InteractiveSession()
  saver.restore(sess, saver_name)
  
  g = tf.get_default_graph()
  x = g.get_tensor_by_name("x-input:0")
  y = g.get_tensor_by_name("y-input:0")
  keep_prob = g.get_tensor_by_name("keep_prob:0")
  accuracy = g.get_tensor_by_name("accuracy:0")
  
  test_accuracy = accuracy.eval(feed_dict={x: test_set.images, y: test_set.labels, keep_prob: 1.0})
  print('Accuracy %g' % (test_accuracy))
  
if __name__ == '__main__':
  tf.app.run()