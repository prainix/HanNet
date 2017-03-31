from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import numpy as np
import os
import re

ttfpath = "../data/ttf/"
fontsuffix = ".TTF"
outpath = "../data/binary/"

fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
#fonts = ["fangsong"]

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
            m = re.search('^uni([ABCDEF\d]+)$', y[1])
            if m is not None:
                hex_str = "0x"+ m.group(1)
                hex_int = int(hex_str, 16)
                if hex_int >= lower_bound and hex_int <= upper_bound:
                    char_list.append(hex_int)    

    data_array = np.empty([len(char_list), pic_size*pic_size])
    for idx, char_val in enumerate(char_list):
        textu = chr(char_val)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        dr.text((0, 0), textu, font=font, fill=0)
        data_array[idx,:] = np.array(im).reshape(1, 28*28)
    
    binary_file = outpath + f
    np.save(binary_file, data_array)
    
    print('Done with font \"{0}\", found {1} valid chars.'.format(f, len(char_list)))