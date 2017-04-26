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

FONTS = PARAMS.fonts
FONT_SIZES = PARAMS.font_sizes
PIC_SIZE = PARAMS.pic_size
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = PARAMS.base_char_count * len(FONTS) * len(FONT_SIZES)
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

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


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+1)*batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+1)*batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images, min(25, images.shape[0]))

  return images, tf.reshape(label_batch, [batch_size])


def generate_binary_files(fonts):
  
  if not os.path.isdir(PARAMS.recordpath):
    os.makedirs(PARAMS.recordpath)

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
  
    filename = os.path.join(PARAMS.recordpath, f + PARAMS.recordsuffix)
    writer = tf.python_io.TFRecordWriter(filename)
    for size in FONT_SIZES:
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


def read_charImages(filename_queue):

  class charImagesRecord(object):
    pass
  result = charImagesRecord()

  reader = tf.TFRecordReader()
  result.key, value = reader.read(filename_queue)
  features = tf.parse_single_example(value, features={
                                     'label': tf.FixedLenFeature([], tf.int64),
                                     'image_raw': tf.FixedLenFeature([], tf.string)})

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  result.uint8image = tf.reshape(image, [PIC_SIZE, PIC_SIZE, 1])

  result.label = features['label']

  return result


def distorted_inputs():
  filenames = [os.path.join(PARAMS.recordpath, f+PARAMS.recordsuffix) for f in FONTS]
  for f in filenames:
    if not tf.gfile.Exists(f):
      generate_binary_files(FONTS)
      break

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_charImages(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d char images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, PARAMS.batch_size,
                                         shuffle=True)
