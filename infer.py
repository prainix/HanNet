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

fonts = ["fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]
font_size = 24

ttfpath = "../../data/ttf/"
fontsuffix = ".TTF"

binarypath = "../data/binary/" + str(font_size) + "pt_all/"
datasuffix = ".npy"

model_dir = "../data/HanNet_CNN"
saver_name = model_dir + "/HanNet"
meta_file = saver_name + ".meta"
  
def main(_):
    saver = tf.train.import_meta_graph(meta_file)
    sess = tf.InteractiveSession()
    saver.restore(sess, saver_name)

    g = tf.get_default_graph()
    x = g.get_tensor_by_name("x-input:0")
    y = g.get_tensor_by_name("y-input:0")
    keep_prob = g.get_tensor_by_name("keep_prob:0")
    accuracy = g.get_tensor_by_name("accuracy:0")
  
    for idx, font in enumerate(fonts):
        binary_file = binarypath + font + datasuffix
        
        image_data = np.load(binary_file)
        image_data = image_data / 255.0
        
        label_data = np.zeros([image_data.shape[0], len(fonts)])
        label_data[:,idx].fill(1)
        
        test_accuracy = accuracy.eval(feed_dict={x: image_data, y: label_data, keep_prob: 1.0})
        #print('Accuracy for font %s is : %g' % (font, test_accuracy))
        print(test_accuracy)
  
if __name__ == '__main__':
    tf.app.run()