import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.inception_score
import tflib.save_images
import tflib.small_imagenet
import tflib.plot
import argparse
import generators
import discriminators

DATA_DIR = '/home/neale/repos/adversarial-toolbox/images/imagenet64'
DIM = 64  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
BATCH_SIZE = 64  # Batch size. Must be a multiple of N_GPUS
ITERS = 300000  # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3  # Number of pixels in each iamge
SAVE_ITERS = [10000, 20000, 50000, 100000, 200000]
SAVE_NAME = "64x64_wgan-gp"

lib.print_model_settings(locals().copy())


def load_args():

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-s', '--save_dir', default='default_dir',type=str, help='experiment_save_dir')
    parser.add_argument('-z', '--z', default=128,type=int, help='noise sample width')
    parser.add_argument('-t', '--tf_trial_name', default=0,type=str, help='tensorboard trial')
    args = parser.parse_args()
    return args


args = load_args()
Generator = generators.GResnetOptim
Discriminator = discriminators.DResnetOptim

with tf.Session() as session:

    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    gen_costs, disc_costs = [], []
    real_data = tf.reshape(2*((tf.cast(real_data_int, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE, DIM, OUTPUT_DIM)
    disc_real = Discriminator(real_data, DIM)
    disc_fake = Discriminator(fake_data, DIM)
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates, DIM), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_costs.append(gen_cost)
    disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs)
    disc_cost = tf.add_n(disc_costs)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    # For generating samples
    samples_100 = Generator(100, DIM, OUTPUT_DIM)
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    n_samples = BATCH_SIZE
    fixed_noise_samples = Generator(n_samples, DIM, OUTPUT_DIM, noise=fixed_noise)

    def generate_image(iteration, random=False):
        if random is True:
            random_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
            random_samples = Generator(n_samples, DIM, OUTPUT_DIM, random_noise)
            samples = session.run(random_samples)
        else:
            samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        samples = samples.reshape((BATCH_SIZE, 3, 64, 64))
        f = 'samples_{}'.format(iteration)
        lib.save_images.save_images(samples, f, args.save_dir)

    def get_inception_score():
        samples = []
        for i in xrange(10):
            samples.append(session.run(samples_100))
        samples = np.concatenate(samples, axis=0)
        samples = ((samples+1.)*(255./2)).astype('int32')
        samples = samples.reshape((-1, 3, 64, 64)).transpose(0, 2, 3, 1)
        return lib.inception_score.get_inception_score(list(samples))

    # Dataset iterator
    train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    # Train loop
    session.run(tf.global_variables_initializer())
    summaries_dir = './board'
    train_writer = tf.summary.FileWriter(summaries_dir + '/train/'+args.save_dir+'/'+args.tf_trial_name,
                                         session.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    saver = tf.train.Saver()
    # saver.restore(session, '../models/gp_wgan/64x64_wgan-gp_100000_steps')
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, random=True)

	if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        if iteration in SAVE_ITERS:
            saver.save(session, 'models/'+SAVE_NAME+'_'+str(iteration)+'_steps')
        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush()

        lib.plot.tick()
