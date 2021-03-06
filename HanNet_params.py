from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import os
import platform

platform_name = platform.system()

if platform_name.lower() == "linux":
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

elif platform_name.lower() == "windows":
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
#fonts = ["Fangsong", "STXINGKA"]
fonts = ["Fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]

# Data size
pic_size = 64
font_sizes = [32, 38, 44, 52, 60, 64]

validation_size = 500
test_ratio = 0.1

# Training Hyper parameters
total_steps = 2000
batch_size = 50
drop_ratio = 0.5

# Hardware
num_threads = 4
use_gpu = True
