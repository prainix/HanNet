"""
HanNet
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import HanNet_input
import HanNet_params as PARAMS

NUM_THREADS = PARAMS.num_threads
FONTS = PARAMS.fonts

def main(_):
    saver = tf.train.import_meta_graph(PARAMS.meta_file)
    sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
    saver.restore(sess, PARAMS.saver_name)

    g = tf.get_default_graph()
    x = g.get_tensor_by_name("x-input:0")
    y = g.get_tensor_by_name("y-input:0")
    keep_prob = g.get_tensor_by_name("keep_prob:0")
    accuracy = g.get_tensor_by_name("accuracy:0")
  
    [test_data, _] = HanNet_input.read_data_sets(False)
    #size = test_data.num_examples()
    test_accuracy = accuracy.eval(feed_dict={'x-input:0': test_data.images[1:2048],
                                             'y-input:0': test_data.labels[1:2048],
                                             'keep_prob:0': 1.0})
    print(test_accuracy)
  
if __name__ == '__main__':
    tf.app.run()
