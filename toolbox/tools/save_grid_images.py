#coding: utf-8
import numpy as np
import os
from glob import glob
import argparse
from scipy.misc import imread, imsave
#import matplotlib.pyplot as plt

i = 0
def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-i', '--image', default=os.getcwd(),type=str, help='PNG image')
  parser.add_argument('-d', '--image_dir', default=None, help='directory where PNG images are stored')

  args = parser.parse_args()
  return args

def deconstruct_dir(args):
    paths = glob(args.image_dir+'/*.png')
    print "{} files to deconstruct".format(len(paths))
    print "{} images".format(len(paths)*128)
    for path in paths:
        args.image = path
        deconstruct_grid(args)

def deconstruct_grid(args):

    f = args.image
    global i
    print f
    images  = []
    x = imread(f)
    print x.shape
    for row in range(x.shape[0]/28):
        for col in range(x.shape[1]/28):
            im = x[28*row:28*(row+1),(col*28):28*(col+1)]
            imsave('./grid_images/im_{}.png'.format(i), im)
            images.append(im)
            i += 1

    x = np.array(images)

if __name__ == '__main__':
    args = load_args()
    if args.image_dir is not None:
        deconstruct_dir(args)
    else:
        deconstruct_grid(args)
