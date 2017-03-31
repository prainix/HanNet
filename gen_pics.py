# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import numpy as np
import os
import re

ttfpath = "../data/ttf/"
fontsuffix = ".TTF"
imgpath = "../data/charimages/"

fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
#fonts = ["fangsong"]

pic_size = 28
font_size = 24

# unicode Chinese chars
lower_bound = 0x4E00
upper_bound = 0x9FA5

for f in fonts:
    fontpath = ttfpath + f + fontsuffix
    exportpath = imgpath + f + "/"
    if not os.path.exists(exportpath):
        os.makedirs(exportpath)

    char_list = []
    ttf = TTFont(fontpath, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            m = re.search('^uni([ABCDEF\d]+)$', y[1])
            if m is not None:
                hex_str = "0x"+ m.group(1)
                hex_int = int(hex_str, 16)
                if hex_int >= lower_bound and hex_int <= upper_bound:
                    char_list.append(hex_int)    

    font = ImageFont.truetype(fontpath, font_size)
    for idx in char_list:
        textu = chr(idx)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        dr.text((0, 0), textu, font=font, fill=0)
        im.save(exportpath + textu + ".png")
    
    print('Done with font \"{0}\", found {1} valid chars.'.format(f, len(char_list)))