from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf

from PIL import Image
from scipy.misc import imresize
from tf_models import InceptionV4Model
from cleverhans.attacks import MomentumIterativeMethod as MIM

slim = tf.contrib.slim
tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
        'input_dir', '../../imgs', 'Input directory with images.')
tf.flags.DEFINE_string(
        'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
        'batch_size', 32, 'How many images process at one time.')
tf.flags.DEFINE_integer(
        'max_images', 5000, 'Number of images to generate')
FLAGS = tf.flags.FLAGS

save_dir = '/home/neale/repos/adversarial-toolbox/images/adversarials/mim/'
save_path = save_dir + 'imagenet/symmetric/inception_v4/'


def save_npy(x_real, x_adv):
    adv_path = save_path+'adv_npy/'
    real_path = save_path +'real_npy/'
    if not os.path.exists(adv_path):
        os.makedirs(adv_path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    for i in range(len(x_adv)):
        np.save(adv_path+'adv_{}.png'.format(i), x_adv[i])
        np.save(real_path+'real_{}.png'.format(i), x_real[i])


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
        with tf.gfile.Open(filepath) as f:
            image = imresize(np.array(Image.open(f).convert('RGB')).astype(np.float), (299, 299, 3)) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def show(real, adv, model, sess, i):
    import matplotlib.pyplot as plt
    pred_real = np.argmax(model.predict(np.expand_dims(real, 0), sess))
    pred_adv = np.argmax(model.predict(np.expand_dims(adv, 0), sess))
    if pred_real == pred_adv:
        return
    norm = np.linalg.norm(real - adv)
    real = (((real + 1.0) * 0.5) * 255.0).astype(np.uint8)
    adv = (((adv + 1.0) * 0.5) * 255.0).astype(np.uint8)
    plt.figure()
    plt.suptitle("attempt " + str(i))
    ax = plt.subplot(1, 3, 1)
    ax.set_title("real: "+str(pred_real))
    plt.imshow(real)
    ax = plt.subplot(1, 3, 2)
    ax.set_title("adv: "+str(pred_adv))
    plt.imshow(adv)
    ax = plt.subplot(1, 3, 3)
    ax.set_title("diff: "+str(norm))
    plt.imshow(real - adv)
    plt.show()


def main(_):
    batch_size = FLAGS.batch_size
    batch_shape = [FLAGS.batch_size, 299, 299, 3]
    with tf.Graph().as_default():
    # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        
        with tf.Session() as sess:
            model = InceptionV4Model(sess=sess)
            model._build()
            mim = MIM(model, back='tf', sess=None)
            mim_params = {'eps_iter': 0.06,
                          'eps': 0.3,
                          'nb_iter': 10,
                          'ord': 2,
                          'decay_factor': 1.0}
            
            x_adv = mim.generate(x_input, **mim_params)
            j = 0
            z_samples = np.zeros((10000, 299, 299, 3))
            real_samples = np.zeros((10000, 299, 299, 3))
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                for (real, adv) in zip(images, adv_images):
                    z_samples[j] = adv
                    real_samples[j] = real
                    # show(real, adv, model, sess, j) 
                    j += 1
                if not (j % 100):
                    print j
                if j >= FLAGS.max_images:
                    print "Max examples exceeded, early stopping"
                    break

            save_npy(real_samples, z_samples)
    return


if __name__ == '__main__':
    tf.app.run()
