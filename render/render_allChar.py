from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import numpy as np
import os
import re

ttfpath = "../../data/ttf/"
fontsuffix = ".TTF"
outpath = "../../data/binary/"

fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]

pic_size = 28
font_size = 24

# unicode Chinese chars
lower_bound = 0x4E00
upper_bound = 0x9FA5

for f in fonts:
    fontpath = ttfpath + f + fontsuffix
    font = ImageFont.truetype(fontpath, font_size)
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)

    char_list = []
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            char_int = y[0]
            if char_int >= lower_bound and char_int <= upper_bound:
                char_list.append(char_int)

    data_array = np.empty([len(char_list), pic_size*pic_size])
    for idx, char_val in enumerate(char_list):
        textu = chr(char_val)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        pos_x = pic_size/2 - font_size/2
        pos_y = pos_x
        dr.text((pos_x, pos_y), textu, font=font, fill=0)
        data_array[idx,:] = np.array(im).reshape(1, pic_size*pic_size)
    
    binary_file = outpath + f
    np.save(binary_file, data_array)
    
    print('Done with font \"{0}\", found {1} valid chars.'.format(f, len(char_list)))