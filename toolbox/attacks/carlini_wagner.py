import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from cw.l0_attack import CarliniL0
from cw.l2_attack import CarliniL2
# from li_attack import CarliniLi

weight_path = '/home/neale/repos/adversarial-toolbox/toolbox/attacks/'


class CWAttack(object):

    def __init__(self, label):
        self.label = label
        self.targets = 1000

    def generate_data(self, data, label, ilsvrc=True):
        inputs = []
        targets = []
        if ilsvrc:
            seq = random.sample(range(1, 1001), 10)
        else:
            seq = range(self.targets)
        for i in seq:
            inputs.append(data)
            targets.append(np.eye(self.targets)[i])

        inputs = np.array(inputs)
        targets = np.array(targets)
        return inputs, targets

    def L0_attack(self, model, img):
        with tf.Session() as sess:
            data = img
            attack = CarliniL0(sess, model, batch_size=1)
            inputs, targets = self.generate_data(data, self.label,
                                                 ilsvrc=True)
            adv = attack.attack(inputs, targets)
            return adv

    def L2_attack(self, model, img):
        with tf.Session() as sess:
            data = img
            attack = CarliniL2(sess, model, batch_size=1,
                               max_iterations=1000)
            inputs, targets = self.generate_data(data, self.label,
                                                 ilsvrc=True)
            print "performing l2 attack"
            adv = attack.attack(inputs, targets)
            return adv

    def Li_attack(self, model, img):
        with tf.Session() as sess:
            data = img
            attack = CarliniL2(sess, model, batch_size=1,
                               max_iterations=1000)
            inputs, targets = self.generate_data(data, self.label,
                                                 ilsvrc=True)
            adv = attack.attack(inputs, targets)
            return adv


class CWModel(object):

    def __init__(self, model, size=224, labels=1000):
        self.num_channels = 3
        self.image_size = 224
        self.num_labels = labels
        fc = model.layers[-2].output
        new_model = Dense(1000, name='fc1000')(fc)
        new_model = Model(model.input, new_model)
        new_model.load_weights(weight_path+'resnet50.h5')
        model.summary()
        self.model = new_model

    def predict(self, x):
        self.model.load_weights(weight_path+'resnet50.h5')
        print self.model(x)
        return self.model(x)


def cw_attack(model, img, label, norm='2'):

    cw = CWAttack(label)
    cw_model = CWModel(model)

    img = img/255.
    if (norm not in ['0', '2', 'i']):
        raise ValueError("Only L0, L2, and Li CW attacks supported")

    if norm == '0':
        adv = cw.L0_attack(cw_model, img)

    if norm == '2':
        adv = cw.L2_attack(cw_model, img)

    if norm == 'i':
        adv = cw.Li_attack(cw_model, img)

    return adv
