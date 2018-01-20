"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf

from PIL import Image
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import inception
from cleverhans.attacks import MomentumIterativeMethod as MIM

slim = tf.contrib.slim
tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
        'checkpoint_path', '../checkpoints/inception_v3.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
        'input_dir', './imgs', 'Input directory with images.')
tf.flags.DEFINE_string(
        'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
        'batch_size', 32, 'How many images process at one time.')
FLAGS = tf.flags.FLAGS

save_dir = '/home/neale/repos/adversarial-toolbox/images/adversarials/mim/'
save_path = save_dir + 'imagenet/symmetric/inception_v3/'


def save_npy(x_real, x_adv):
    adv_path = save_path+'adv_npy/'
    real_path = save_path +'real_npy/'
    for i in range(len(adv_path)):
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


class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.img = tf.placeholder(tf.float32, (None, 299, 299, 3))

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=False,
                    reuse=reuse)
            self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

def freeze_graph(sess, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants 
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(
            keep_var_names or []))
        output_names = output_names or []
        #output_names += [v.op.name for v in tf.global_variables()]
        output_names += [v.name for v in tf.get_default_graph().as_graph_def().node]
        input_graph_def = graph.as_graph_def()
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, 
                        output_names, freeze_var_names)
        tf.train.write_graph(frozen_graph, './', 'inception_v3.pb', as_text=False)


def main(_):
    batch_size = FLAGS.batch_size
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    targeted = False
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        model = InceptionModel(num_classes)
        with tf.Session() as sess:
            predictions = model(x_input)
            mim = MIM(model, back='tf', sess=None)
            mim_params = {'eps_iter': 0.06,
                          'eps': 0.3,
                          'nb_iter': 10,
                          'ord': 2,
                          'decay_factor': 1.0}

            x_adv = mim.generate(x_input, **mim_params)
            sys.exit(0)
            saver = tf.train.Saver(slim.get_model_variables())

            saver.restore(sess, FLAGS.checkpoint_path)
            z_samples = np.zeros((10000, 299, 299, 3))
            real_samples = np.zeros((10000, 299, 299, 3))
	    meta_graph_def = tf.train.export_meta_graph(
	    	filename='tmp/imagenet/inception_v3.meta')
	    saver.save(sess, 'tmp/imagenet/inception_v3.ckpt')
            #freeze_graph(sess)
            sys.exit(0)
            """
            a = [n.name for n in tf.get_default_graph().as_graph_def().node]
            for item in a:
                print item
            """
            j = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                for (real, adv) in zip(images, adv_images):
                    z_samples[j] = adv
                    real_samples[j] = real
                    j += 1
                if not (j % 100):
                    print j
                if j >= 5000:
                    print "Max examples exceeded, early stopping"
                    break

            save_npy(real_samples, z_samples)


if __name__ == '__main__':
    tf.app.run()
