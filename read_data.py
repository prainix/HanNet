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

def read_data_sets(with_validation):
    binarypath = "../data/binary/"
    datasuffix = ".npy"
    #fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
    fonts = ["STHUPO", "STLITI", "STXINGKA", "STXINWEI"]

    all_data = np.array([])
    for idx, font in enumerate(fonts):
        binary_file = binarypath + font + datasuffix
        
        image_data = np.load(binary_file)
        image_data = image_data / 255.0
        
        label_data = np.zeros([image_data.shape[0], len(fonts)])
        label_data[:,idx].fill(1)
        
        comb_data = np.hstack((image_data,label_data))
        
        if all_data.size == 0:
            all_data = comb_data
        else:
            all_data = np.vstack((all_data, comb_data))
    
    np.random.shuffle(all_data)

    # last elements of each row is the label
    images, labels = np.hsplit(all_data, [all_data.shape[1]-len(fonts)])

    validation_size = 500
    test_size = 5000

    train_images = images[test_size:]
    train_labels = labels[test_size:]
    test_images = images[:test_size]
    test_labels = labels[:test_size]
    
    if(with_validation):
        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]
    
    train = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)

    if(with_validation):
        validation = DataSet(validation_images, validation_labels)
        return train, validation, test
    else:
        return train, test
