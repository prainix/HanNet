# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw
#from scipy.misc import imread, imsave, imresize
import numpy as np

fontpath = "../data/ttf/STXINGKA.TTF"
exportpath = "../data/charimages/"
size = 40

start,end = (0x4E00, 0x9FA5) # unicode Chinese chars
font = ImageFont.truetype(fontpath, size)

# # a char not included in test font
# text0 = chr(0x5209)
# im0 = Image.new("L", (size, size), 255)
# dr0 = ImageDraw.Draw(im0)
# dr0.text((0, 0), text0, font=font, fill=0)

for idx in range(start, end):
    textu = chr(idx)
    im = Image.new("L", (size, size), 255)
    dr = ImageDraw.Draw(im)
    dr.text((0, 0), textu, font=font, fill=0)
    # if(not (np.array(im) == np.array(im0)).all()):
    #     im.save(exportpath + textu + ".png")
    im.save(exportpath + textu + ".png")