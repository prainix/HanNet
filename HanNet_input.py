from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from bisect import bisect_left

import numpy as np
import HanNet_params as PARAMS
import os

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

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
  fonts = PARAMS.fonts
  num_fonts = len(PARAMS.fonts)

  for f in fonts:
    binary_file = PARAMS.binarypath + f + PARAMS.datasuffix
    if not os.path.isfile(binary_file):
      generate_binary_files(fonts)
      break

  all_data = np.array([])
  for idx, f in enumerate(fonts):
    binary_file = PARAMS.binarypath + f + PARAMS.datasuffix

    image_data = np.load(binary_file)
    image_data = image_data / 255.0
    
    label_data = np.zeros([image_data.shape[0], num_fonts])
    label_data[:,idx].fill(1)
    
    comb_data = np.hstack((image_data,label_data))
    
    if all_data.size == 0:
      all_data = comb_data
    else:
      all_data = np.vstack((all_data, comb_data))
  
  np.random.shuffle(all_data)

  # last num_fonts elements of each row is the label
  images, labels = np.hsplit(all_data, [all_data.shape[1]-num_fonts])

  validation_size = PARAMS.validation_size
  test_size = int(all_data.shape[0] * PARAMS.test_ratio)
  test_size = min(test_size, 2048)
  train_size = all_data.shape[0] - test_size

  train_images = images[test_size:]
  train_labels = labels[test_size:]
  test_images = images[:test_size]
  test_labels = labels[:test_size]
  
  if(with_validation):
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    train_size = train_size - validation_size
  
  train = DataSet(train_images, train_labels)
  test = DataSet(test_images, test_labels)

  print('Training set size: {0}'.format(train_size))
  print('Test set size: {0}'.format(test_size))
  if(with_validation):
    print('Validation set size: {0}'.format(validation_size))
    validation = DataSet(validation_images, validation_labels)
    return train, validation, test
  else:
    return train, test

def generate_binary_files(fonts):
  pic_size = PARAMS.pic_size
  font_sizes = PARAMS.font_sizes
  
  if not os.path.isdir(PARAMS.binarypath):
    os.makedirs(PARAMS.binarypath)

  baseChar_list = []
  fontpath = PARAMS.ttfpath + PARAMS.base_font + PARAMS.fontsuffix
  ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
  for x in ttf["cmap"].tables:
    for y in x.cmap.items():
      char_int = y[0]
      if char_int >= PARAMS.lower_bound and char_int <= PARAMS.upper_bound:
        baseChar_list.append(char_int)   
  baseChar_list.sort()

  for f in fonts:
    fontpath = PARAMS.ttfpath + f + PARAMS.fontsuffix
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
  
    char_list = []
    for x in ttf["cmap"].tables:
      for y in x.cmap.items():
        char_int = y[0]
        if (index(baseChar_list, char_int) != -1):
          char_list.append(char_int)
  
    data_array = np.zeros([len(char_list)*len(font_sizes), pic_size*pic_size])
    for i, size in enumerate(font_sizes):
      font = ImageFont.truetype(fontpath, size)
      for j, char_val in enumerate(char_list):
        textu = chr(char_val)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        pos_x = pic_size/2 - size/2
        pos_y = pos_x
        dr.text((pos_x, pos_y), textu, font=font, fill=0)
        data_array[i*len(char_list)+j,:] = np.array(im).reshape(1, pic_size*pic_size)
    
    out_name = PARAMS.binarypath + f 
    np.save(out_name, data_array)
    
    print('Done generating data for font \"{0}\", found {1} valid base chars.'.format(f, len(char_list)))
