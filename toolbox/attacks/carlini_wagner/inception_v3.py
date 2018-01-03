import tensorflow as tf
import numpy as np
import os
import time
import random
import scipy.misc
import matplotlib.pyplot as plt

from PIL import Image
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
# from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
    'checkpoint_path', './../cleverhans/weights/inception_v3/inception_v3.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_targets(n_samples, n_targets=1001):
    """
    Generate the input labels to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    targets = []
    for i in range(n_samples):
        seq = random.sample(range(1, n_targets), 100)
        for j in seq:
            targets.append(np.eye(n_targets)[j])

    targets = np.array(targets)

    return targets


class InceptionModel(object):

    """Model class for CleverHans library."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_labels = num_classes
        self.built = False
        self.num_channels = 3
        self.image_size = 299

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

    def predict(self, x):
        return self.__call__(x)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

      Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

      Yields:
        filenames: list file names without path of each image
          Lenght of this list could be less than batch_size, in this case only
          first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
    """Saves images to the output directory.

    Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def plot(x, x_adv):
    """ plots adversarial image and distortion """
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.subplot(1, 3, 2)
    plt.imshow(x_adv)
    diff = x - x_adv
    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.show()


if __name__ == "__main__":

    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    with tf.Session() as sess:
        # data, model =  MNIST(), MNISTModel("models/mnist", sess)
        # data, model = CIFAR(), CIFARModel("models/cifar", sess)
        model = InceptionModel(num_classes)
        attack = CarliniL2(sess, model, batch_size=FLAGS.batch_size, max_iterations=1000,
                           confidence=0)

        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.checkpoint_path,
                master=FLAGS.master)
        saver.restore(sess, FLAGS.checkpoint_path)
        sess.run(tf.global_variables_initializer())

        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            targets = load_targets(FLAGS.batch_size)
            timestart = time.time()
            adv = attack.attack(images, targets)
            timeend = time.time()

            print timeend-timestart, "seconds to run ", len(adv), "samples"

            for i in range(len(adv)):
                print("Classification:", model.model.predict(adv[i:i+1]))
                print("Total distortion:", np.sum((adv[i]-images[i])**2)**.5)
                plot(images[i], adv[i])
            save_images(adv, filenames, FLAGS.output_dir)
