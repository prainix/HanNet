from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

class DataSet(object):

  def __init__(self,
               images,
               labels):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets():
  # ori_array is a [6764*4, 28*28+4] array
  ori_array = np.loadtxt("four.out")
  np.random.shuffle(ori_array)
  # last 4 elements of each row form the label array
  images, labels = np.hsplit(ori_array,[ori_array.shape[1]-4])
  validation_size = 2000
  test_size = 5500
  # train_images = images[test_size:]
  # train_labels = labels[test_size:]
  # test_images = images[:test_size]
  # test_labels = labels[:test_size]
  # validation_images = train_images[:validation_size]
  # validation_labels = train_labels[:validation_size]
  # train_images = train_images[validation_size:]
  # train_labels = train_labels[validation_size:]
  train_images = images[test_size:]
  train_labels = labels[test_size:]
  test_images = images[:test_size]
  test_labels = labels[:test_size]

  train = DataSet(train_images, train_labels)
  # validation = DataSet(validation_images,
  #                      validation_labels)
  test = DataSet(test_images, test_labels)
  # return train, validation, test
  return train, test
