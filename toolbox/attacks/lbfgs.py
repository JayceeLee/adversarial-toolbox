import os
import foolbox
import keras
import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from tools.cifar_base import cifar_model
from foolbox.criteria import TargetClass
from keras.backend.tensorflow_backend import set_session


def config_tf(gpu=0.5):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu
    set_session(tf.Session(config=config))
    keras.backend.set_learning_phase(0)


def display_im(img, title=None):

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def load_model(weights=None, top=None):
    kmodel = cifar_model(top=top, path=weights)
    # kmodel.summary()
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))
    return fmodel


def get_target(label, r):

    target = label
    while target == label:
        target = np.random.randint(r)
    return target


def print_stats(adversarial, model):

    failed = 0
    l1 = np.argmax(model.predictions(adversarial.image))
    l2 = np.argmax(model.predictions(adversarial.original_image))
    print "Adversarial Predicted: ", l1
    print "Original Predicted: ", l2
    if l1 == l2:
        failed = 1
    return failed


def score_dataset(model, x, label):

    misses = 0
    for img in x:
        pred = np.argmax(model.predictions(img))
        if pred != label:
            misses += 1
            print pred, label
    return misses


def generate(model, image, targets):

    label = np.argmax(model.predictions(image))
    print "label: ", label
    if targets == 2 and label == 0:
        print "classified real as adversarial"
        return None
    target_class = get_target(label, targets)
    print "Target Class: {}".format(target_class)
    # criterion = TargetClass(target_class)
    try:
        attack = foolbox.attacks.LBFGSAttack(model)  # , criterion)
        adversarial = attack(image, label, unpack=False)
        return adversarial
    except:
        print "FAILED"
        return None
