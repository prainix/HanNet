from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import os

username = getpass.getuser()
home_dir = os.environ['HOME']

project_basedir = home_dir + "/HanNet/"
cloud_mountdir = home_dir + "/Cloud-HanNet-Shared/"

ttfpath = cloud_mountdir + "fonts_raw/"
fontsuffix = ".TTF"

binarypath = project_basedir + "data/binary/"
datasuffix = ".npy"

model_dir = project_basedir + "model/"
saver_name = model_dir + "HanNet"
meta_file = saver_name + ".meta"

# Fonts selection
lower_bound = 0x4E00
upper_bound = 0x9FA5
base_font = "STHUPO"
fonts = ["Fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]
num_fonts = 8

# Data size
pic_size = 64
font_sizes = [64]

validation_size = 500
test_size = 1000

# Training control
batch_size = 100
