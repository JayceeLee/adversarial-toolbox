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

    dirs = ['/home/neale/repos/adversarial-toolbox/toolkit/unpacked_cifar_train',
            '/home/neale/repos/adversarial-toolbox/gp-wgan/rand_grid_images',
            '/home/neale/repos/adversarial-toolbox/toolkit/images/jsma/trial_single_advgeneric_p0'
            ]
    mag_all = []
    for d in dirs:
        images = glob(d+'/*.png')
        gx, gy, mag = [], [], []
        print len(images)
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
            gx.append(imx)
            gy.append(imx)
            mag.append(magnitude)
        mag_all.append(mag)

    mag1 = mag_all[0]
    mag2 = mag_all[1]
    mag3 = mag_all[2]

    plot_pdf(mag1, mag2, mag3)

def plot_pdf(d1, d2):
    from scipy.stats import norm
    from scipy.stats import gaussian_kde
    print "{} items in d1, \n{} items in d2".format(len(d1), len(d2))
    # Plotting histograms and PDFs
    d1 = np.array(d1).flatten()
    d2 = np.array(d2).flatten()
    #d3 = np.array(d3).flatten()
    """
    plt.figure()
    density = gaussian_kde(d2)
    x2 = np.linspace(min(d2), max(d2), len(d2))
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(x2,density(x2))
    plt.show()
    """
    plt.figure()
    x = np.linspace(min(d1), max(d1), len(d1))
    x2 = np.linspace(min(d2), max(d2), len(d2))
    #x3 = np.linspace(min(d3), max(d3), len(d3))
    print max(d2), min(d2)
    plt.plot(x, norm.pdf(x, np.mean(d1), np.std(d1)),'g-', lw=3, alpha=0.9, label='learned pdf gen_real')
    plt.plot(x2, norm.pdf(x2, np.mean(d2), np.std(d2)),'c-', lw=3, alpha=0.9, label='learned pdf adversarial')
    #plt.plot(x3, norm.pdf(x3, np.mean(d3), np.std(d3)),'m-', lw=3, alpha=0.9, label='norm pdf adversarial')
    plt.xlabel('magnitude')
    plt.suptitle("pdf of gradients on all images")
    #plt.hist(d1, bins=15, normed=True, stacked=True, cumulative=True)
    #plt.hist(d2, bins=15, normed=True, stacked=True, cumulative=True)
    plt.title("mag all samples")
    plt.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    args = load_args()
    #x, y, z, img = collect_grad(args)
    #plot_hist(x, y, z, img)
    #plot_sobel(img)
    plot_sobel_hist(args)
