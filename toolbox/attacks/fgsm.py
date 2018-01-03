from __future__ import absolute_import

import numpy as np
import keras
import tensorflow as tf

from cleverhans.utils_tf import model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

params_home = '/home/neale/repos/adversarial-toolbox/toolbox/attacks/resnet50.h5'

class FGSM(object):

    def mifgsm(model, img, label, targets):

    keras.layers.core.K.set_learning_phase(0)
    tf.set_random_seed(1234)
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Use label smoothing on one_hot
    y = np.array([label])
    z = np.zeros((len(y), 1000))
    z[np.arange(len(y)), y] = 1
    assert z.shape[1] == 1000, z.shape
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 1000))

    print "Defined TensorFlow model graph."
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    wrap.model.load_weights(params_home)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 255.}
    adv_x = fgsm.generate_np(img, **fgsm_params)

    return adv_x


