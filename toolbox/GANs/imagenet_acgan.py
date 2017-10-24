
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
import tflib.tiny_imagenet
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets
import matplotlib.pyplot as plt
import time
import functools
import locale

locale.setlocale(locale.LC_ALL, '')

DATA_DIR = '/home/neale/repos/adversarial-toolbox/images/imagenet200'

DIM = 256
BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 200000  # How many iterations to train for
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 64*64*3  # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4  # Initial learning rate
DECAY = True  # Whether to decay LR over learning
N_CRITIC = 5  # Critic steps per generator steps
INCEPTION_FREQUENCY = 500  # How frequently to calculate Inception score

CONDITIONAL = True  # Whether to train a conditional or unconditional model
ACGAN = True  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.2  # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

lib.print_model_settings(locals().copy())

Generator = generators.GImResnet
Discriminator = discriminators.DImResnet

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    real_data = tf.reshape(2*((tf.cast(real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    fake_data = Generator(BATCH_SIZE, DIM, OUTPUT_DIM, real_labels)

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []

    real_and_fake_data = tf.concat([
        real_data,
        fake_data,
    ], axis=0)
    real_and_fake_labels = tf.concat([
        real_labels,
        real_labels
    ], axis=0)

    disc_all, disc_all_acgan = Discriminator(real_and_fake_data,
                                             DIM, real_and_fake_labels,
                                             CONDITIONAL, ACGAN)
    disc_real = disc_all[:BATCH_SIZE]
    disc_fake = disc_all[BATCH_SIZE:]

    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

    if CONDITIONAL and ACGAN:
        disc_acgan_costs.append(tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=disc_all_acgan[:BATCH_SIZE],
                labels=real_and_fake_labels[:BATCH_SIZE])
        ))
        disc_acgan_accs.append(tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(
                        tf.argmax(disc_all_acgan[:BATCH_SIZE], dimension=1)),
                    real_and_fake_labels[:BATCH_SIZE]
                ),
                tf.float32
            )
        ))
        disc_acgan_fake_accs.append(tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(
                        tf.argmax(disc_all_acgan[BATCH_SIZE:], dimension=1)),
                    real_and_fake_labels[BATCH_SIZE:]
                ),
                tf.float32
            )
        ))

    labels = real_labels
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, DIM, labels, CONDITIONAL, ACGAN)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
    disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs)
        disc_acgan_acc = tf.add_n(disc_acgan_accs)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_costs = []
    gen_acgan_costs = []
    n_samples = GEN_BS_MULTIPLE * BATCH_SIZE
    fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
    if CONDITIONAL and ACGAN:
        disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples, DIM, OUTPUT_DIM, fake_labels), DIM, fake_labels, CONDITIONAL, ACGAN)
        gen_costs.append(-tf.reduce_mean(disc_fake))
        gen_acgan_costs.append(tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
        ))
    else:
        gen_costs.append(-tf.reduce_mean(Discriminator(
                                            Generator(n_samples, DIM, OUTPUT_DIM, fake_labels),
                                            DIM, fake_labels, CONDITIONAL, ACGAN
                                            )[0]))
    gen_cost = (tf.add_n(gen_costs))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs)))

    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, DIM, OUTPUT_DIM, fixed_labels, noise=fixed_noise)

    def generate_image(i, random=False):
        if random is True:
            random_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
            random_samples = Generator(100, DIM, OUTPUT_DIM, fixed_labels, noise=random_noise)
            samples = session.run(random_samples)
        else:
            samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 64, 64)), 'samples_{}'.format(i), 'ac-gan_adv')

    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, DIM, OUTPUT_DIM, fake_labels_100)


    def get_inception_score(n):
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 64, 64)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.tiny_imagenet.load(BATCH_SIZE, DATA_DIR)

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

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration})

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
            _data = np.reshape(_data, (BATCH_SIZE, OUTPUT_DIM))
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run([disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op], feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan', _disc_acgan)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _labels in dev_gen():
                images = np.reshape(images, (BATCH_SIZE, OUTPUT_DIM))
                _dev_disc_cost = session.run([disc_cost], feed_dict={real_data_int: images,real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, random=False)
            # generate_image(iteration, random=True)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()
