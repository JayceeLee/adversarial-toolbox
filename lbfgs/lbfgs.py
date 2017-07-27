import foolbox
from foolbox import Adversarial
from foolbox.attacks import LBFGSAttack, DeepFoolAttack
from foolbox.criteria import TargetClass, Misclassification, OriginalClassProbability
from foolbox.distances import MeanSquaredDistance
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imread, imsave, imresize
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch as torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
from PIL import Image
import os
import sys

net = models.resnet50(pretrained=True)
# Switch to evaluation mode
net.eval()
successful_images = 0
bad_dims = 0

keras.backend.set_learning_phase(0)
#model = ResNet50(weights='imagenet')
#preprocessing = (np.array([104, 116, 123]), 1)
#fmodel = foolbox.models.KerasModel(model, bounds=(0, 255))#, preprocessing=preprocessing)
model = foolbox.models.PyTorchModel(net, bounds=(0, 1000), num_classes=1000)

paths = glob('../images/imagenet12/*.JPEG')
for path in paths:

    target_class = np.random.randint(1000)
    #x = imread(path)
    #x = imresize(x, (224, 224))
    #x = np.expand_dims(x, axis=0)

    im_orig = Image.open(path)
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    x = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)])(im_orig)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = x.cuda()
        net = net.cuda()
        print ("using GPU")
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:1000]
    label = I[0]
    print '\n', label, '\n'
    # Remove the mean
    print path, '\n'
    #x = x[0]
    x = imresize(im_orig, (224, 224, 3))
    criterion = TargetClass(target_class)
    #criterion = Misclassification()
    #criterion = OriginalClassProbability(0.5)
    attack = LBFGSAttack(model, criterion)
    #attack = DeepFoolAttack(fmodel, criterion)
    #print type(x), Variable(x).data.cpu().numpy().shape
    #x = Variable(x).data.cpu().numpy()
    print np.max(x)
    print x.shape, type(x)
    plt.imshow(x)
    plt.show()
    print x.shape
    distance = MeanSquaredDistance
    adv = Adversarial(model, criterion, x, label, distance=distance)
    attack(adv)
    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                             transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])

    image = tf(adv.image)
    #imsave('../images/resnet50_lbfgs_adv/im_{}-{}.png'.format(label, new_label), adversarial)
    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.subplot(1, 3, 3)
    plt.imshow(image-adv.original_image)
    plt.show()
