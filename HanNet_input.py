from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from bisect import bisect_left

import numpy as np
import tensorflow as tf
import HanNet_params as PARAMS
import os
import random

FONTS = PARAMS.fonts
NUM_FONTS = len(FONTS)
PIC_SIZE = PARAMS.pic_size

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _generate_image_and_label_batch(read_input_list, min_queue_examples, batch_size, shuffle):
  if shuffle:
    images, label_batch = tf.train.shuffle_batch_join(
        read_input_list,
        batch_size=batch_size,
        capacity=min_queue_examples + batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch_join(
        read_input_list,
        batch_size=batch_size,
        capacity=min_queue_examples + batch_size)

  return images, tf.reshape(label_batch, [batch_size])


def generate_record_files(fonts, is_eval):
  
  if is_eval is True:
    recordpath = PARAMS.recordpath_eval
    font_sizes = PARAMS.font_sizes_eval
  else:
    recordpath = PARAMS.recordpath_train
    font_sizes = PARAMS.font_sizes_train

  if not os.path.isdir(recordpath):
    os.makedirs(recordpath)

  baseChar_list = []
  fontpath = PARAMS.ttfpath + PARAMS.base_font + PARAMS.fontsuffix
  ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
  for x in ttf["cmap"].tables:
    for y in x.cmap.items():
      char_int = y[0]
      if char_int >= PARAMS.lower_bound and char_int <= PARAMS.upper_bound:
        baseChar_list.append(char_int)   
  baseChar_list.sort()

  for label, f in enumerate(fonts):
    fontpath = PARAMS.ttfpath + f + PARAMS.fontsuffix
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
  
    char_list = []
    for x in ttf["cmap"].tables:
      for y in x.cmap.items():
        char_int = y[0]
        if (index(baseChar_list, char_int) != -1):
          char_list.append(char_int)
  
    filename = os.path.join(recordpath, f + PARAMS.recordsuffix)
    writer = tf.python_io.TFRecordWriter(filename)
    for size in font_sizes:
      font = ImageFont.truetype(fontpath, size)
      for char_val in char_list:
        textu = chr(char_val)
        im = Image.new("L", (PIC_SIZE, PIC_SIZE), 255)
        dr = ImageDraw.Draw(im)
        pos_x = PIC_SIZE/2 - size/2
        pos_y = pos_x
        dr.text((pos_x, pos_y), textu, font=font, fill=0)
        image = np.array(im).reshape(1, PIC_SIZE*PIC_SIZE)
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'label': _int64_feature(label),
                                   'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    
    print('Done generating data for font \"{0}\", found {1} valid base chars.'.format(f, len(char_list)))


def read_and_process(filename_queue, distort):

  reader = tf.TFRecordReader()
  _, value = reader.read(filename_queue)
  features = tf.parse_single_example(value, features={
                                     'label': tf.FixedLenFeature([], tf.int64),
                                     'image_raw': tf.FixedLenFeature([], tf.string)})

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  uint8image = tf.reshape(image, [PIC_SIZE, PIC_SIZE, 1])

  label = features['label']

  float_image = tf.cast(uint8image, tf.float32)

  if distort is True:
    float_image = tf.image.random_flip_left_right(float_image)
    angle = tf.random_uniform([1], minval=-3.14/8, maxval=3.14/8, dtype=tf.float32)
    float_image = tf.contrib.image.rotate(float_image, angle)

  norm_image = float_image / 255.0

  return norm_image, label


def inputs(is_eval, distort, shuffle):
  if is_eval is True:
    filenames = [os.path.join(PARAMS.recordpath_eval, f+PARAMS.recordsuffix) for f in FONTS]
    epochs_limit = 1
    examples_per_epoch = PARAMS.base_char_count*NUM_FONTS*len(PARAMS.font_sizes_eval)
  else:
    filenames = [os.path.join(PARAMS.recordpath_train, f+PARAMS.recordsuffix) for f in FONTS]
    epochs_limit = PARAMS.epochs_limit
    examples_per_epoch = PARAMS.base_char_count*NUM_FONTS*len(PARAMS.font_sizes_train)

  min_queue_examples = int(examples_per_epoch*PARAMS.min_fraction_of_examples_in_queue)

  for f in filenames:
    if not tf.gfile.Exists(f):
      generate_record_files(FONTS, is_eval)
      break

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs_limit)

  # Read examples from files in the filename queue.
  read_input_list = [read_and_process(filename_queue, distort=distort)
                     for _ in range(NUM_FONTS)]

  print ('Filling queue with %d char images.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input_list, min_queue_examples,
                                         PARAMS.batch_size, shuffle=shuffle)
