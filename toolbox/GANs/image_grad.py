import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage import color
import argparse
from scipy.ndimage import filters
from glob import glob
# load image

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-i', '--image', default=os.getcwd(),type=str, help='PNG image')
  parser.add_argument('-d', '--image_dir', default=os.getcwd(),type=str, help='directory where PNG images are stored')

  args = parser.parse_args()
  return args


def collect_grad(args):
    img = imread(args.image)
    img = np.array(img)
    print "image =", img
    # compute gradient of image
    gx, gy, gz = np.gradient(img)
    print "gx =", gx
    print "gy =", gy
    print "gz =", gz
    return gx, gy, gz, img

def plot_grads(x, y, z, img):

    plt.close("all")
    plt.figure()
    plt.suptitle("Image, and it gradient along each axis")
    ax = plt.subplot("141")
    ax.axis("off")
    ax.imshow(img)
    ax.set_title("image")

    ax = plt.subplot("142")
    ax.axis("off")
    ax.imshow(x)
    ax.set_title("gx")

    ax = plt.subplot("143")
    ax.axis("off")
    ax.imshow(y)
    ax.set_title("gy")

    ax = plt.subplot("144")
    ax.axis("off")
    ax.imshow(z)
    ax.set_title("gz")
    plt.show()

def plot_hist(x, y, z, img):
    plt.figure()
    plt.suptitle("image, histogram of gradients on each axis")
    mean_gx = np.mean((x[:, :, 0], x[:,:,1], x[:,:,2]), axis=0)
    mean_gy = np.mean((y[:, :, 0], y[:,:,1], y[:,:,2]), axis=0)
    mean_gz = np.mean((z[:, :, 0], z[:,:,1], z[:,:,2]), axis=0)
    ax = plt.subplot("141")
    ax.imshow(img)
    ax.set_title("image")

    ax = plt.subplot("142")
    ax.hist(mean_gx, bins='auto')
    ax.set_title("gx")

    ax = plt.subplot("143")
    ax.hist(mean_gy, bins='auto')
    ax.set_title("gy")

    ax = plt.subplot("144")
    ax.hist(mean_gz, bins='auto')
    ax.set_title("gz")

    plt.show()

def plot_sobel(im):

    # plot grayscale
    im = color.rgb2gray(im)
    # sobel derivative filters
    imx = np.zeros(im.shape)
    filters.sobel(im,1,imx)

    imy = np.zeros(im.shape)
    filters.sobel(im,0,imy)

    magnitude = np.sqrt(imx**2+imy**2)
    plt.figure()
    plt.suptitle("gradients on x, y, and magnitude")
    ax = plt.subplot("141")
    ax.imshow(im)
    ax.set_title("image")
    ax = plt.subplot("142")
    ax.imshow(imx)
    ax.set_title("gx")
    ax = plt.subplot("143")
    ax.imshow(imy)
    ax.set_title("gy")
    ax = plt.subplot("144")
    ax.imshow(magnitude)
    ax.set_title("mag")
    plt.show()

def plot_sobel_hist(args, im=None):

    images = glob(args.image_dir+'/*.png')
    # set shape for first example
    im = imread(images[0])
    im = np.array(im)
    im = color.rgb2gray(im)
    # sobel derivative filters
    imx = np.zeros(im.shape)
    filters.sobel(im,1,imx)
    imy = np.zeros(im.shape)
    filters.sobel(im,0,imy)
    magnitude = np.sqrt(imx**2+imy**2)
    gx = np.zeros(imx.shape)
    gy = np.zeros(imy.shape)
    mag = np.zeros(magnitude.shape)

    for image in images:
        im = imread(image)
        im = np.array(im)
        # plot grayscale
        im = color.rgb2gray(im)
        # sobel derivative filters
        imx = np.zeros(im.shape)
        filters.sobel(im,1,imx)
        imy = np.zeros(im.shape)
        filters.sobel(im,0,imy)
        magnitude = np.sqrt(imx**2+imy**2)
        gx += imx
        gy += imx
        mag += magnitude

    imx = gx/len(images)
    imy = gy/len(images)
    magnitude = mag/len(images)
    print imx.shape
    print imy.shape
    print magnitude.shape
    plt.figure()
    #plt.suptitle("gradients on x, y, and magnitude")
    plt.suptitle("mean gradients on x, y, and magnitude")
    #ax = plt.subplot("141")
    #ax.imshow(im)
    #ax.set_title("image")
    ax = plt.subplot("142")
    n, bins, patches = ax.hist(imx, normed=1, histtype='bar')
    ax.set_title("gx")
    ax = plt.subplot("143")
    n, bins, patches = ax.hist(imy, normed=1, histtype='bar')
    ax.set_title("gy")
    ax = plt.subplot("144")
    n, bins, patches = ax.hist(magnitude, normed=1, histtype='bar')
    ax.set_title("mag")
    plt.show()


if __name__ == '__main__':
    args = load_args()
    #x, y, z, img = collect_grad(args)
    #plot_hist(x, y, z, img)
    #plot_sobel(img)
    plot_sobel_hist(args)
