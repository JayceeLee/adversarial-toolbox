"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division

import os

from cleverhans.attacks import MomentumIterativeMethod as MIM
import numpy as np
from PIL import Image
from load_data import load_images_tf, save_images_tf
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
        'checkpoint_path', './weights/inception_v3/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
        'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
        'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
        'batch_size', 32, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS



def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    batch_size = FLAGS.batch_size
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    targeted = False
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
    # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        model = InceptionModel(num_classes)
        with tf.Session() as sess:

            mim = MIM(model, back='tf', sess=None)
            mim_params = {'eps_iter': 0.06,
                          'eps': 0.3,
                          'nb_iter': 10,
                          'ord': 2,
                          'decay_factor': 1.0}


            x_adv = mim.generate(x_input, **mim_params)

            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                    scaffold=tf.train.Scaffold(saver=saver),
                    checkpoint_filename_with_path=FLAGS.checkpoint_path,
                    master=FLAGS.master)
            saver.restore(sess, FLAGS.checkpoint_path)
            sess.run(tf.global_variables_initializer())
            # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            i = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                print "input images: ", images.shape
                #adv_images = cw.generate_np(images, **cw_params)
                i += 16
                print i
                # print filenames
                # print adv_images.shape
                # adv_images = cw.generate_np(
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
