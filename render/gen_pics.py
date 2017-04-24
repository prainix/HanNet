# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import numpy as np
import os
import re

ttfpath = "../../data/ttf/"
fontsuffix = ".TTF"
imgpath = "../../data/charimages/"

#fonts = ["fangsong", "Kaiti", "MicrosoftYahei", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI","STZHONGS"]
#fonts = ["fangsong", "Kaiti", "SimHei", "SimSun", "STHUPO", "STLITI", "STXINGKA", "STXINWEI"]
fonts = ["STXINWEI"]

pic_size = 64
font_size = 64

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
            char_int = y[0]
            if char_int >= lower_bound and char_int <= upper_bound:
                char_list.append(char_int)

    font = ImageFont.truetype(fontpath, font_size)
    for idx in char_list:
        textu = chr(idx)
        im = Image.new("L", (pic_size, pic_size), 255)
        dr = ImageDraw.Draw(im)
        pos_x = pic_size/2 - font_size/2
        pos_y = pos_x
        dr.text((pos_x, pos_y), textu, font=font, fill=0)
        #im = im.rotate(15, expand=1)
        im.save(exportpath + textu + ".png")
    
    print('Done with font \"{0}\", found {1} valid chars.'.format(f, len(char_list)))