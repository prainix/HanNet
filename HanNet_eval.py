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
import math
import sys
from datetime import datetime

import HanNet_input
import HanNet_train
import HanNet_params as PARAMS

BATCH_SIZE  = PARAMS.batch_size
MODEL_DIR   = PARAMS.model_dir
NUM_THREADS = PARAMS.num_threads
USE_GPU     = PARAMS.use_gpu

def main(_):

  if not tf.gfile.Exists(MODEL_DIR):
    print("Model dir does not exist!")
    sys.exit(-1)

  if USE_GPU is True:
    dev = '/gpu:0'
  else:
    dev = '/cpu:0'

  # read data using queue
  images, labels = HanNet_input.inputs(is_eval=True, distort=False, shuffle=False)
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  with tf.device(dev):
    logits = HanNet_train.vgg_lite(images, keep_prob)

  # Calculate predictions.
  top_k_op = tf.nn.in_top_k(logits, labels, 1)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(MODEL_DIR + '/eval')

  saver = tf.train.Saver()
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                                        allow_soft_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    tf.local_variables_initializer().run()
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

      true_count = 0  # Counts the number of correct predictions.
      step = 0
      while not coord.should_stop():
        predictions = sess.run([top_k_op], feed_dict={'keep_prob:0': 1.0})
        true_count += np.sum(predictions)
        step += 1
        if not step % 50:
          precision = true_count / (step*BATCH_SIZE)
          print('%d: precision = %.3f' % (step, precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    # Compute precision @ 1.
    total_sample_count = step * BATCH_SIZE
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Precision @ 1', simple_value=precision)

  
if __name__ == '__main__':
    tf.app.run()
