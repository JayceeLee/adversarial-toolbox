import os, sys
sys.path.append(os.getcwd())

import time
import functools
import argparse

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.inception_score
import tflib.ops.layernorm
import tflib.plot

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
DATA_DIR = '/home/neale/repos/adversarial-toolbox/images/imagenet64'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 300000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
SAVE_ITERS = [10000, 20000, 50000, 100000, 200000]
SAVE_NAME = "64x64_wgan-gp"

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-s', '--save_dir', default='default_dir',type=str, help='experiment_save_dir')
  parser.add_argument('-z', '--z', default=128,type=int, help='noise sample width')
  parser.add_argument('-t', '--tf_trial_name', default=0,type=str, help='tensorboard trial')
  args = parser.parse_args()
  return args

args = load_args()
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    # 101-layer ResNet G and D
    return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators

def ResnetGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in xrange(6):
        output = ResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = ResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

# For calculating inception score
samples_100 = Generator(100)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 64, 64)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    summaries_dir = './board'
    train_writer = tf.summary.FileWriter(summaries_dir + '/train/'+args.save_dir+'/'+args.tf_trial_name,
                                         session.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    # Train loop

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES))

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE / len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'samples_{}.png'.format(iteration), './samples/64x64/trial1')
    """
    def generate_random_image(iteration):
        random_noise_z = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        random_noise_samples_z = Generator(args.z, noise=random_noise_z)
        samples = session.run(random_noise_samples_z)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'samples_{}'.format(frame), save, individual=False)
    """
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('/home/neale/repos/adversarial-toolbox/gp-wgan/models/64x64_wgan-gp_100000_steps.meta')
    saver.restore(session, '/home/neale/repos/adversarial-toolbox/gp-wgan/models/64x64_wgan-gp_100000_steps')

    # Dataset iterator
    train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    gen = inf_train_gen()
    """
    import sys
    for iteration in xrange(500):
        generate_random_image(iteration)
        sys.exit(0)
    """
    for iteration in xrange(ITERS):

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

	print "plot train disc cost"
        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: _data})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration)

        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush()

        if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        if iteration in SAVE_ITERS:
            saver.save(session, 'models/'+SAVE_NAME+'_'+str(iteration)+'_steps')
        lib.plot.tick()
