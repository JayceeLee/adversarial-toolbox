import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.mnist_custom
import tflib.plot

import generators
import discriminators

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
SAVE_DIR = '../../images/wgan_samples/MNIST_ADV'
SAVE_ITERS = [1000, 5000, 10000]
TF_TRIAL = 0
START = 10000
SAVE_NAME = 'MNIST_ADV'
MODEL_DIR = '../models/weights/gp_wgan/MNIST_ADV/'
MODEL_NAME = 'MNIST_ADV_10000_steps'

Generator = generators.GMnist
Discriminator = discriminators.DMnist

lib.print_model_settings(locals().copy())

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE, DIM, OUTPUT_DIM)

disc_real = Discriminator(real_data, DIM)
disc_fake = Discriminator(fake_data, DIM)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')


if MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, DIM), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real,
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, DIM, OUTPUT_DIM, noise=fixed_noise)

"""
def generate_image(it):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)),
        'samples_{}.png'.format(it), SAVE_DIR
    )
"""
def generate_image(iteration, random=False):
        if random is True:
            random_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
            random_samples = Generator(128, DIM, OUTPUT_DIM, random_noise)
            samples = session.run(random_samples)
        else:
            samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        samples = samples.reshape((128, 28, 28))
        f = 'samples_{}.png'.format(iteration)
        lib.save_images.save_images(samples, f, SAVE_DIR)


# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)

# train_gen, dev_gen, test_gen = lib.mnist_custom.load(BATCH_SIZE)


def inf_train_gen():
    while True:
        for images, labels in train_gen():
            yield images


# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    train_writer = tf.summary.FileWriter('./board/MNIST/train/'+str(TF_TRIAL),
                                         session.graph)
    test_writer = tf.summary.FileWriter('board/MNIST/test')

    saver = tf.train.Saver()
    saver.restore(session, MODEL_DIR+MODEL_NAME)

    gen = inf_train_gen()

    for iteration in xrange(START, ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost,
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, random=False)
            generate_image(iteration+1, random=True)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        if iteration in SAVE_ITERS:
            saver.save(session, MODEL_DIR+SAVE_NAME+'_'+str(iteration)+'_steps')
        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush()

        lib.plot.tick()
