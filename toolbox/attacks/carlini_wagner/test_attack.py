# test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.
import os
import sys
import tensorflow as tf
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from imageio import imsave, imread

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prepare_data(image, target, targeted=True, start=0):
    inputs = []
    targets = []
    if targeted:
        seq = random.sample(range(1, 1001), 1)
        for j in seq:
            inputs.append(image)
            print target.shape
            targets.append(np.eye(target.shape[0])[j])
    else:
        inputs.append(image)
        targets.append(target)

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def generate_data(data, samples, targeted=True, start=0, inception=False):
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 100)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = ImageNet(), InceptionModel(sess)
        attack = CarliniL2(sess, model, batch_size=1, max_iterations=10000,
                           confidence=0)

        timestart = time.time()
        nb_iterations = data.test_data.shape[0]
        print "Starting attack on {} images".format(nb_iterations)
        start = 0
        for i in range(start, nb_iterations+start):
            target_image = data.test_data[i]
            target_label = data.test_labels[i]
            image, target = prepare_data(target_image, target_label,
                                         targeted=True, start=i)
            print "target: ", np.argmax(target)
            print "*******************************"
            print "starting image ", i
            print "*******************************"
            adv = attack.attack(image, target)
            timeend = time.time()

            image = np.reshape(image, (299, 299, 3)).astype(np.uint8)
            adv = np.reshape(adv, (299, 299, 3)).astype(np.uint8)
            print adv
            print adv.shape
            print("Took", timeend-timestart, "seconds to run", '1', "samples.")
            print("Total distortion:", np.sum((adv-image)**2)**.5)
            imsave(os.getcwd()+'/../../../images/adversarials/cw/imagenet/symmetric/l2/inception_v3/adv/1Kadv_{}.png'.format(i), adv)
            imsave(os.getcwd()+'/../../../images/adversarials/cw/imagenet/symmetric/l2/inception_v3/real/1Kim_{}.png'.format(i), image)
            sys.exit(0)
            # adv = (adv+.5) * 255
            # print("Classification:", model.model.predict(adv[i:i+1]))
            # x1 = (adv + .5) * 255.

            # print("Classification Real-Adv: ", model.predict(image, pred=True, sess=sess), '- ', model.predict(adv, pred=True, sess=sess))
