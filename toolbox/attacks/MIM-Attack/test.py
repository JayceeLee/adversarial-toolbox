import numpy as np
import tensorflow as tf
from scipy.misc import imread, imshow
from config_inception_v3 import InceptionV3Model
from config_inception_v4 import InceptionV4Model
from config_inception_resnet_v2 import InceptionResNetModel
from config_resnet_v2_101 import ResNetV2Model
slim = tf.contrib.slim
# x = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/adv_npy/adv_0.png.npy')
x = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/adv_npy/adv_1.png.npy')
# y = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/real_npy/real_0.png.npy')
y = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/real_npy/real_1.png.npy')

print "nice, loaded"

y = np.expand_dims(y, 0).astype(np.float32)
x = np.expand_dims(x, 0).astype(np.float32)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    print "Inception V3"
    model = InceptionV3Model(sess)
    model._build()
    for i in range(1):
        print np.argmax(model.predict(y))
        print np.argmax(model.predict(x))
    model._free()

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    print "Inception V4"
    model = InceptionV4Model(sess)
    model._build()
    for i in range(1):
        print np.argmax(model.predict(y))
        print np.argmax(model.predict(x))
    model._free()

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    print "Inception ResNet v2"
    model = InceptionResNetModel(sess)
    model._build()
    for i in range(1):
        print np.argmax(model.predict(y))
        print np.argmax(model.predict(x))
    model._free()
    
tf.reset_default_graph()
with tf.Session(config=config) as sess:
    print "ResNet V2"
    model = ResNetV2Model(sess)
    model._build()
    for i in range(1):
        print np.argmax(model.predict(y))
        print np.argmax(model.predict(x))
    model._free()

imshow(x[0])
