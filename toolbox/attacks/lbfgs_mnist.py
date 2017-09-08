import sys
import math
import foolbox
from foolbox.criteria import TargetClass, OriginalClassProbability
import keras
import numpy

from foolbox.utils import softmax
import mnist
import matplotlib.pyplot as plt
import scipy.misc
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
# instantiate model
keras.backend.set_learning_phase(0)
kmodel = mnist.mnist(top='vanilla')
kmodel.summary()

fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))
num_gen = 0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
save_dir = '../images/adversarials/lbfgs/mnist/vanilla/'

def sigmoid(x):
      return 1 / (1 + math.exp(-x))

correct = 0
for i in range(204, len(x_test)):
    image = x_test[i]
    print "Testing Image ", i, "\t", image.shape
    if num_gen >= 60000:
        print "\n\nFinshed!\n"
        sys.exit(0)
    # get source image and label
    label = numpy.argmax(fmodel.predictions(image))
    target_class = label
    while target_class is label:
        target_class = numpy.random.randint(10)

    print "Original Predicted: ", label
    print "Target Class: {}".format(target_class)

    criterion = TargetClass(target_class)
    attack = foolbox.attacks.LBFGSAttack(fmodel, criterion)

    #plt.imshow(numpy.reshape(image, (28,28)), cmap='gray')
    #plt.show()

    try:
        adversarial = attack(image, label, unpack=False)
        adv = adversarial.image
        adv = numpy.reshape(adversarial.image, (28, 28))
    except:
        print "FAILED for ", i

        continue

    nz = numpy.count_nonzero(adv)
    print "Nonzero Pixels: {}".format(nz)
    new_label = numpy.argmax(fmodel.predictions(adversarial.image))
    print "New Prediction: ", new_label
    if label == new_label:
        print "FAILED with img: {}".format(i)
    else:
        num_gen += 1
        scipy.misc.imsave(save_dir+'{}_{}.png'.format(new_label, num_gen), adv)
        print "SUCCESS, images saved: {}".format(num_gen)

    print '------------'
