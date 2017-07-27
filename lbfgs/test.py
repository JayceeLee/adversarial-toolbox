import foolbox
import keras
import numpy
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import scipy.misc
# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (numpy.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
image = scipy.misc.imread('bee.jpg')
image = scipy.misc.imresize(image, (224, 224))
label = numpy.argmax(fmodel.predictions(image))

# apply attack on source image
attack  = foolbox.attacks.LBFGSAttack(fmodel)
adversarial = attack(image[:,:,::-1], label)

plt.subplot(1, 3, 1)
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.imshow(adversarial)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - image)
plt.show()
