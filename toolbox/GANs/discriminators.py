import sys
sys.path.append('../')
from tflib.ops.conv2d import Conv2D
from tflib.ops.linear import Linear
from tflib.ops.res import ResidualBlock, prelu
from tflib.ops.res import OptimizedResBlockDisc1
from tflib.ops.res import LeakyReLU, nonlinearity
from models.cifar_base import cifar_model
import keras.backend as K
import tensorflow as tf

def InitDetector():

    detector = cifar_model(top='gan')  # init with a sigmoid
    detector.load_weights('./models/lbfgs_cifar.h5', by_name=True)
    for layer in detector.layers:
        layer.trainable=False
    K.set_learning_phase(0)
    return detector

def DCifar(inputs, dim):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = Conv2D('D1.1', 3, dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('D1.2', dim, 2*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('D1.3', 2*dim, 4*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    # we relax the dimensionality constraint here. Make it big
    output = tf.reshape(output, [-1, 4*4*4*dim])
    output = Linear('D1.Output', 4*4*4*dim, 1, output)
    return tf.reshape(output, [-1])


def DResnet(inputs, dim, dim_out):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

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
    output = Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])


def DCifarResnet(inputs, labels, dim, conditional=True, acgan=True):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1('Discriminator.1', output, dim)
    output = ResidualBlock('Discriminator.2', dim, dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', dim, dim, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', dim, dim, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = Linear('Discriminator.Output', dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if conditional and acgan:
        output_acgan = Linear('Discriminator.ACGANOutput', dim, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None



def PatchGAN(targets, inputs, dim):
    inputs = tf.reshape(inputs, [-1, 3, 32, 32])
    targets = tf.reshape(targets, [-1, 3, 32, 32])
    data = tf.concat([inputs, targets], axis=1)
    print "\ndata shape: ", data.shape
    disc1 = OptimizedResBlockDisc1('Discriminator.0', data, dim, input_dim=6, stride=1)  # conv down :)
    disc1 = prelu(disc1, 'Discriminator.1.prelu')
    print "shape after disc block 1: {} -> {}".format(data.shape, disc1.shape)
    disc2 = ResidualBlock('Discriminator.2', dim, dim, 3, disc1, resample='down')
    print "shape after disc block 2: {} -> {}".format(disc1.shape, disc2.shape)
    disc3 = ResidualBlock('Discriminator.3', dim, dim, 3, disc2, resample='down')
    # output = ResidualBlock('Discriminator.4', dim, dim, 3, output, resample='down')
    print "shape after disc block 3: {} -> {}".format(disc2.shape, disc3.shape)
    output = prelu(disc3, 'output.prelu')
    output = tf.reduce_mean(output, axis=[2,3])

    return output


def DImResnet(inputs, dim, labels, conditional=True, acgan=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = OptimizedResBlockDisc1(output, dim)
    output = ResidualBlock('Discriminator.2', dim, dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', dim, dim, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', dim, dim, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', dim, dim, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = Linear('Discriminator.Output', dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if conditional and acgan:
        output_acgan = Linear('Discriminator.ACGANOutput', dim, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


def DResnetOptim(inputs, dim, labels=None, conditional=True, acgan=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down', labels=labels)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    # output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = Linear('Discriminator.Output', 4*4*8*dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])

    if conditional and acgan:
        output_acgan = Linear('Discriminator.ACGANOutput', 4*4*8*dim, 1000, output)
        return output_wgan, output_acgan

    return output_wgan, None


def DMnist(inputs, dim):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = Conv2D('Discriminator.1',1,dim,5,output,stride=2)
    output = LeakyReLU(output)
    output = Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 4*4*4*dim])
    output = Linear('Discriminator.Output', 4*4*4*dim, 1, output)

    return tf.reshape(output, [-1])
