# -*- coding: utf-8 -*-
from PIL import Image, ImageFont, ImageDraw
#from scipy.misc import imread, imsave, imresize
import numpy as np

fontpath = ["../data/ttf/STXINGKfinal_array.TTF", "../data/ttf/STLITI.TTF", "../data/ttf/STXINWEI.TTF","../data/ttf/STHUPO.TTF"]
exportpath = "../data/ttf/charimages/"
flag = 1
size = 28
start,end = (0x4E00, 0x9FA5) # unicode Chinese chars

for ft in fontpath:
    font = ImageFont.truetype(ft, size)
    # a char not included in test font
    text0 = chr(0x5209)
    im0 = Image.new("L", (size, size), 255)
    dr0 = ImageDraw.Draw(im0)
    dr0.text((0, 0), text0, font=font, fill=0)
    if(flag):
        final_array = np.array(im0).reshape(1, 28*28)/255.0
        flag = 0
    else:
        final_array = np.concatenate((final_array, np.array(im0).reshape(1, 28*28)/255.0), axis=0)
    for idx in range(start, end):
        textu = chr(idx)
        im = Image.new("L", (size, size), 255)
        dr = ImageDraw.Draw(im)
        dr.text((0, 0), textu, font=font, fill=0)
        if(not (np.array(im) == np.array(im0)).all()):
            final_array = np.concatenate((final_array, np.array(im).reshape(1, 28*28)/255.0), axis=0)
print(final_array.shape)

label_array = np.vstack((np.tile(np.array([1,0,0,0]), (final_array.shape[0]/4,1)),
       np.tile(np.array([0,1,0,0]), (final_array.shape[0]/4,1)),
       np.tile(np.array([0,0,1,0]), (final_array.shape[0]/4,1)),
       np.tile(np.array([0,0,0,1]), (final_array.shape[0]/4,1))))
print(label_array.shape)
f = np.hstack((final_array,label_array))
np.savetxt('four.out', f, fmt='%.8f')
print(f.shape)