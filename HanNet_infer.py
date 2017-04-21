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

import HanNet_params as PARAMS

NUM_THREADS = PARAMS.num_threads

def main(_):
    saver = tf.train.import_meta_graph(PARAMS.meta_file)
    sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
    saver.restore(sess, PARAMS.saver_name)

    g = tf.get_default_graph()
    x = g.get_tensor_by_name("x-input:0")
    y = g.get_tensor_by_name("y-input:0")
    keep_prob = g.get_tensor_by_name("keep_prob:0")
    accuracy = g.get_tensor_by_name("accuracy:0")
  
    for idx, font in enumerate(fonts):
        binary_file = PARAMS.binarypath + font + PARAMS.datasuffix
        
        image_data = np.load(binary_file)
        image_data = image_data / 255.0
        
        label_data = np.zeros([image_data.shape[0], len(fonts)])
        label_data[:,idx].fill(1)
        
        test_accuracy = accuracy.eval(feed_dict={x: image_data, y: label_data, keep_prob: 1.0})
        print('Accuracy for font %s is : %g' % (font, test_accuracy))
        print(test_accuracy)
  
if __name__ == '__main__':
    tf.app.run()
