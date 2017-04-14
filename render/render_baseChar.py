from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from bisect import bisect_left

import numpy as np
import os
import re

ttfpath = "../../data/ttf/"
fontsuffix = ".TTF"
outpath = "../../data/binary/24pt_base/"

#fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
base_font = "STHUPO"
fonts = ["fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]

pic_size = 28
font_size = 24

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
font = ImageFont.truetype(fontpath, font_size)
ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
for x in ttf["cmap"].tables:
    for y in x.cmap.items():
        char_int = y[0]
        if char_int >= lower_bound and char_int <= upper_bound:
            baseChar_list.append(char_int)   
baseChar_list.sort()

for f in fonts:
    fontpath = ttfpath + f + fontsuffix
    font = ImageFont.truetype(fontpath, font_size)
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)

    char_list = []
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            char_int = y[0]
            if (index(baseChar_list, char_int) != -1):
                char_list.append(char_int)

    data_array = np.empty([len(char_list), pic_size*pic_size])
    for idx, char_val in enumerate(char_list):
        textu = chr(char_val)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        dr.text((0, 0), textu, font=font, fill=0)
        data_array[idx,:] = np.array(im).reshape(1, pic_size*pic_size)
    
    binary_file = outpath + f
    np.save(binary_file, data_array)
    
    print('Done with font \"{0}\", found {1} valid base chars.'.format(f, len(char_list)))