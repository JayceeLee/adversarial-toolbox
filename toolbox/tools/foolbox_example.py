import sys
import keras
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
img_path = './elephant.jpg'
x = imread(img_path)

plt.imshow(x)
plt.show()
x = imresize(x, (224, 224, 3))

x = x.astype(np.float32)

label = np.argmax(fmodel.predictions(x))
print "original: ", label
print "target: 111"

criterion = foolbox.criteria.TargetClass(111)
attack = foolbox.attacks.LBFGSAttack(fmodel, criterion)
adversarial = attack(x[:, :, ::-1], label, maxiter=5)

adv_label = np.argmax(fmodel.predictions(adversarial))
x_adv = adversarial[:, :, ::-1]

imsave('./adv_elephant.png', x_adv)
x_valid = imread('./adv_elephant.png')

x_valid = x_valid.astype(np.float32)
valid_label = np.argmax(fmodel.predictions(x_valid))

x = preprocess_input(np.expand_dims(x_valid, 0))
keras = np.argmax(kmodel.predict(x))


print adv_label, valid_label, keras

