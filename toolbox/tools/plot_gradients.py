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
#
    from scipy.misc import imresize

    dirs = ['/home/neale/repos/adversarial-toolbox/images/cifar/train',
            #'/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/resnet50'
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn',
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn1',
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn2',
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn3',
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn4',
            '/home/neale/repos/adversarial-toolbox/images/adversarials/lbfgs/cifar/vggbn5'
            ]
    mag_all, labels = [], []
    for i, d in enumerate(dirs):
        if d[-11:] == 'imagenet12/':
            images = glob(d+'*.JPEG')
        else:
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
        if i == 0: labels.append("real samples")
        else: labels.append("lbfgs iter {}".format(i))
        mag_all.append(mag)
    import gc
    gc.collect()
    plot_pdf(mag_all, labels)

def plot_pdf(data, labels):
    # Plotting histograms and PDFs
    from scipy.stats import norm
    import matplotlib.cm as cm
    import random
    colors = cm.rainbow(np.linspace(0, 1, len(data)))

    plt.figure()
    for i, c in zip(range(len(data)), colors):
        g = np.array(data[i]).flatten()
        x = np.linspace(min(g), max(g), len(g))
        plt.plot(x, norm.pdf(x, np.mean(g), np.std(g)), color=c, lw=3, alpha=0.9, label=labels[i])

    plt.title("mag all samples")
    plt.suptitle("pdf of gradients on all images")
    plt.xlabel('magnitude')
    plt.legend(loc='best', frameon=False)
    plt.show()
    plt.savefig('imagenet_grad.png')

if __name__ == '__main__':
    args = load_args()
    #x, y, z, img = collect_grad(args)
    #plot_hist(x, y, z, img)
    #plot_sobel(img)
    plot_sobel_hist(args)
