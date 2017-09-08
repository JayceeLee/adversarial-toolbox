import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from scipy.misc import imsave, imread
import sys
from glob import glob

net = models.vgg16(pretrained=True)
save_adv = '/home/neale/repos/adversarial-toolbox/images/adv-vgg16-nolabel/'
save_adv_label = '/home/neale/repos/adversarial-toolbox/images/adv-vgg16-label/'
image_dir = '/home/neale/repos/adversarial-toolbox/images/imagenet12/'
# Switch to evaluation mode
net.eval()
successful_images = 152
bad_dims = 0
paths = glob(image_dir+'*.JPEG')
print "creating adversarials for 5000 images"
for idx, path in enumerate(paths):

    if successful_images > 5000:
        sys.exit(0)

    im_orig = Image.open(path)
    # im_orig = Image.open('/datasets2/ILSVRC2012/train/n01440764/n01440764_30045.JPEG')

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]


    # Remove the mean
    im = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)
    try:
        r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
    except:
        bad_dims += 1
        print ("incorrectly sized images: {}".format(bad_dims))
        print("error image : {}".format(path))
        continue

    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    str_label_orig = labels[np.int(label_orig)].split(',')[0]
    str_label_pert = labels[np.int(label_pert)].split(',')[0]
    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)

    if str_label_orig == str_label_pert:
        print ("failed on image {} - {}".format(idx, str_label_pert))
        continue
    else:
        successful_images += 1
        print ("successful images generated: {}".format(successful_images))


    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                            transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                            transforms.ToPILImage(),
                            transforms.CenterCrop(224)])

    image = np.clip(tf(pert_image.cpu()[0]), 0, 255)
    imsave(save_adv+"adv_vgg16_{}.png".format(idx, image), image)

