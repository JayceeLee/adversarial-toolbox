from __future__ import absolute_import
from __future__ import division

import os
import math
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imshow, imread

from SRGAN.lib.ops import *
from SRGAN.lib.model import data_loader, generator, SRGAN
from SRGAN.lib.model import test_data_loader, inference_data_loader
from SRGAN.lib.model import save_images, SRResnet

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('mode', 'inference', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', 'SRGAN/pretrained/model-200000', '')
Flags.DEFINE_boolean('pre_trained_model', True, 'true: global_step=0, else: ckpt')
Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'SRGAN or SRResnet')
Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
Flags.DEFINE_string('vgg_ckpt', 'SRGAN/vgg_19.ckpt', 'path to checkpoint file')
Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')

# The data preparing operation
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the high resolution input data')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue')
Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue')
Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue')
# Generator configuration
Flags.DEFINE_integer('num_resblock', 16, 'residual blocks in the generator')
# The content loss parameter
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature in perceptual loss')
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'ratio between content and adversarial loss')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'scaling factor for perceptual loss if using vgg')
# The training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')

FLAGS = Flags.FLAGS


def superres(x):
    if FLAGS.flip == True:
        FLAGS.flip = False

    x = x / np.max(x)
    with tf.device('/device:GPU:0'):
        inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3],
                                    name='inputs_raw')

        with tf.variable_scope('generator'):
            if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                gen_output = generator(inputs_raw, 3, reuse=tf.AUTO_REUSE,
                                       FLAGS=FLAGS)
            else:
                raise NotImplementedError('Unknown task!!')


        with tf.name_scope('convert_image'):
            inputs = deprocessLR(inputs_raw)
            outputs = deprocess(gen_output)
            converted_inputs = tf.image.convert_image_dtype(inputs, 
                                                            dtype=tf.uint8, 
                                                            saturate=True)
            converted_outputs = tf.image.convert_image_dtype(outputs, 
                                                             dtype=tf.uint16,
                                                             saturate=True)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='generator')
        weight_initiallizer = tf.train.Saver(var_list)

        init_op = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Load the pretrained model
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        j = 0
        input_im = np.array([x]).astype(np.float32)
        # imshow(input_im[0])
        # print input_im.shape
        results, outputs = sess.run([converted_outputs, outputs], 
                           feed_dict={inputs_raw: input_im})

        return results

"""
if __name__ == '__main__':
    import sys

    img = imread(sys.argv[1])
    print sys.argv[1]
    imshow(img)
    img = superres(img)
    print img
    imshow(img[0])
"""
