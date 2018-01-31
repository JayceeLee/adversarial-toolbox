# Boilerplate imports.
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
slim=tf.contrib.slim
import tf_models

import saliency

def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  im = ((im + 1) * 127.5).astype(np.uint8)
  P.imshow(im)
  P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowDivergingImage(grad, title='', percentile=99, ax=None):  
  if ax is None:
    fig, ax = P.subplots()
  else:
    fig = ax.figure
  
  P.axis('off')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
  fig.colorbar(im, cax=cax, orientation='vertical')
  P.title(title)

def LoadImage(file_path):
  im = PIL.Image.open(file_path)
  im = np.asarray(im)
  return im / 127.5 - 1.0

images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
sess = tf.Session()
model = tf_models.InceptionV3Model(sess)
model._build(sal=True)

# Construct the scalar neuron tensor.
logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
neuron_selector = tf.placeholder(tf.int32)
y = logits[0][neuron_selector]
# Construct tensor for predictions.
prediction = tf.argmax(logits, 1)
print prediction
dir = '../../../images/adversarials/mim/imagenet/symmetric/inception_v3/'
img1 = np.load(dir + 'adv_npy/adv_3.png.npy')
img2 = np.load(dir + 'real_npy/real_3.png.npy')
#img1 = (img1-np.min(img1))/(np.max(img1)-np.min(img1)) 
#img2 = (img2-np.min(img2))/(np.max(img2)-np.min(img2)) 
# Show the image
im = img2
ShowImage(im)

# Make a prediction. 
# prediction_class = sess.run(prediction, feed_dict = {images: [im]})[0]
prediction_class = model.predict(np.expand_dims(im, 0))

print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 237
