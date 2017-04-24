from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from bisect import bisect_left

import numpy as np
import os
import re

pic_size = 28

ttfpath = "../../data/ttf/"
fontsuffix = ".TTF"
outpath = "../../data/binary/base_allsize/"

#fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
base_font = "STHUPO"
fonts = ["fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]
font_sizes = [20, 22, 24, 26, 28]

# unicode Chinese chars
lower_bound = 0x4E00
upper_bound = 0x9FA5

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

baseChar_list = []
fontpath = ttfpath + base_font + fontsuffix
ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
for x in ttf["cmap"].tables:
    for y in x.cmap.items():
        char_int = y[0]
        if char_int >= lower_bound and char_int <= upper_bound:
            baseChar_list.append(char_int)   
baseChar_list.sort()

for f in fonts:
    fontpath = ttfpath + f + fontsuffix
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)

    char_list = []
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            char_int = y[0]
            if (index(baseChar_list, char_int) != -1):
                char_list.append(char_int)

    data_array = np.empty([len(char_list)*len(font_sizes), pic_size*pic_size])
    for i, size in enumerate(font_sizes):
        font = ImageFont.truetype(fontpath, size)
        for j, char_val in enumerate(char_list):
            textu = chr(char_val)
            im = Image.new("L", (pic_size, pic_size), 255)
            dr = ImageDraw.Draw(im)
            pos_x = pic_size/2 - font_size/2
            pos_y = pos_x
            dr.text((pos_x, pos_y), textu, font=font, fill=0)
            data_array[i*len(font_sizes)+j,:] = np.array(im).reshape(1, pic_size*pic_size)
    
    binary_file = outpath + f
    np.save(binary_file, data_array)
    
    print('Done with font \"{0}\", found {1} valid base chars.'.format(f, len(char_list)))