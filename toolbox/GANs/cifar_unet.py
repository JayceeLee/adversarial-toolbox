
import os, sys
sys.path.append(os.getcwd())

import generators
import discriminators
import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets
import scipy.misc
import time
import functools
import locale
from glob import glob

locale.setlocale(locale.LC_ALL, '')

DATA_DIR = 'images/cifar-10-batches-py'

DIM = 128
BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 200000  # How many iterations to train for
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4  # Initial learning rate
DECAY = True  # Whether to decay LR over learning
N_CRITIC = 5  # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000  # How frequently to calculate Inception score

lib.print_model_settings(locals().copy())

Generator = generators.UNet
Discriminator = discriminators.PatchGAN
with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 32, 32, 3])
    real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    #real_data = tf.reshape(2*((tf.cast(real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    #real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    real_data = tf.cast(real_data_int, tf.float32)/256. - .5
    real_data += tf.random_uniform(shape=[BATCH_SIZE, 32, 32, 3],minval=0.,maxval=1./128) # dequantize
    fake_data = Generator(real_data, DIM, OUTPUT_DIM)

    disc_costs = []

    real_and_fake_data = tf.concat([
        real_data,
        fake_data,
    ], axis=0)
    real_and_fake_labels = tf.concat([
        real_labels,
        real_labels
    ], axis=0)

    disc_all = Discriminator(real_data, fake_data, DIM)
    disc_real = disc_all[:BATCH_SIZE]
    disc_fake = disc_all[BATCH_SIZE:]

    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

    labels = real_labels
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(real_data, interpolates, DIM)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
    disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs)
    disc_cost = disc_wgan
    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_costs = []
    gen_costs.append(-tf.reduce_mean(Discriminator(real_data, Generator(real_data, DIM, OUTPUT_DIM), DIM)[0]))
    gen_cost = (tf.add_n(gen_costs))
    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples

    def generate_image(i, images):
        fixed_samples = Generator(images, DIM, OUTPUT_DIM)
        samples = session.run(fixed_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'samples_{}'.format(i), 'unet_samples')

    # Function for calculating inception score
    samples_100 = Generator(100, DIM, OUTPUT_DIM)

    def get_inception_score(n, images):
        samples_100 = Generator(images, DIM, OUTPUT_DIM)
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)

    def inf_train_gen():
        while True:
            for images, labels in train_gen():
                yield images, labels

    for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g is None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    dev_image_paths = glob(DATA_DIR+'/val/*.JPEG')
    rand_idxs = np.random.randint(0, len(dev_image_paths), 100)
    test_images = np.empty((len(rand_idxs), 32, 32, 3))
    for idx, i in enumerate(rand_idxs):
        test_images[idx] = scipy.misc.imread(dev_image_paths[i], mode='RGB')

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration})

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000, test_images)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _labels in dev_gen():
                _dev_disc_cost = session.run([disc_cost], feed_dict={real_data_int: images,real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, test_images)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()
