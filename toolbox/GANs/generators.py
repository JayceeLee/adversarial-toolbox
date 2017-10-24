
import tensorflow as tf
from tflib.ops.res import ResidualBlock, BottleneckResidualBlock
from tflib.ops.res import OptimizedResBlockDisc1
from tflib.ops.res import Normalize, NormalizeD, nonlinearity
from tflib.ops.linear import Linear
from tflib.ops.batchnorm import Batchnorm
from tflib.ops.conv2d import Conv2D
from tflib.ops.deconv2d_unet import Deconv2D_Unet
from tflib.ops.deconv2d import Deconv2D
from tflib.ops.res import UResBlock, prelu

def GCifar(n_samples, dim, dim_out, n, noise=None):

    if noise is None:
        noise = tf.random_normal([n_samples, n])
    output = Linear('G.Input', n, 4*4*4*dim, noise)
    output = Batchnorm('G.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*dim, 4, 4])
    output = Deconv2D('G.2', 4*dim, 2*dim, 5, output)
    output = Batchnorm('G.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Deconv2D('G.3', 2*dim, dim, 5, output)
    output = Batchnorm('G.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Deconv2D('G.5', dim, 3, 5, output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])


def GCifarResnet(n_samples, dim, dim_out, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.reshape(output, [-1, dim, 4, 4])
    output = ResidualBlock('Generator.1', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', dim, dim, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = Conv2D('Generator.Output', dim, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])


def GImResnet(n_samples, dim, dim_out, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.reshape(output, [-1, dim, 4, 4])
    output = ResidualBlock('Generator.1', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.4', dim, dim, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = Conv2D('Generator.Output', dim, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])


def GResnetOptim(n_samples, dim, dim_out, labels, noise=None, nonlinearity=tf.nn.relu):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', labels=labels)
    #output = ResidualBlock('Generator.Res5', 1*dim, 1*dim, 3, output, resample='up')

    output = NormalizeD('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])

def GResnet(n_samples, dim, dim_out, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, dim_out])

def UNet(samples, dim, dim_out):

    enc = tf.reshape(samples, [-1, 3, 32, 32])
    enc1 = Conv2D('Generator.1.enc', 3, dim, 4, enc, he_init=False, stride=2)  # 16x16
    enc1 = nonlinearity(enc1)
    enc1 = Normalize('Generator.1.enc.N', enc1)
    enc2 = Conv2D('Generator.2.enc', dim, dim*2, 4, enc1, he_init=False, stride=2)  # 8x8
    enc2 = nonlinearity(enc2)
    enc2 = Normalize('Generator.2.enc.N', enc2)
    enc3 = Conv2D('Generator.3.enc', dim*2, dim*4, 4, enc2, he_init=False, stride=2)  # 4x4
    enc3 = nonlinearity(enc3)
    enc3 = Normalize('Generator.3.enc.N', enc3)
    enc4 = Conv2D('Generator.4.enc', dim*4, dim*8, 4, enc3, he_init=False, stride=2)  # 2x2
    enc4 = nonlinearity(enc4)
    enc4 = Normalize('Generator.4.enc.N', enc4)
    enc5 = Conv2D('Generator.5.enc', dim*8, dim*8, 4, enc4, he_init=False, stride=2)  #  1x1
    enc5 = nonlinearity(enc5)
    enc5 = Normalize('Generator.5.enc.N', enc5)
    print enc5.shape

    dec1 = nonlinearity(enc5)
    print "begin: ", dec1.shape
    dec1 = Deconv2D_Unet('Generator.1.dec', 8*dim, 8*dim, 4, dec1)
    print "1: ", dec1.shape
    dec1 = Normalize('Generator.1.dec.N', dec1)
    dec1 = tf.nn.dropout(dec1, keep_prob=0.5)

    dec2 = tf.concat([dec1, enc4], axis=1)
    dec2 = nonlinearity(dec2)
    dec2 = Deconv2D_Unet('Generator.2.dec', 8*2*dim, 8*dim, 4, dec2)
    print "2: ", dec2.shape
    dec2 = Normalize('Generator.2.dec.N', dec2)
    dec2 = tf.nn.dropout(dec2, keep_prob=0.5)
    dec3 = tf.concat([dec2, enc3], axis=1)
    dec3 = nonlinearity(dec3)
    dec3 = Deconv2D_Unet('Generator.3.dec', 8*dim, 4*dim, 4, dec3)
    print "3: ", dec3.shape
    dec3 = Normalize('Generator.3.dec.N', dec3)
    dec3 = tf.nn.dropout(dec3, keep_prob=0.5)

    dec4 = tf.concat([dec3, enc2], axis=1)
    dec4 = nonlinearity(dec4)
    dec4 = Deconv2D_Unet('Generator.4.dec', 4*dim, 2*dim, 4, dec4)
    print "4: ", dec4.shape
    dec4 = Normalize('Generator.4.dec.N', dec4)

    dec5 = tf.concat([dec4, enc1], axis=1)
    dec5 = nonlinearity(dec5)
    dec5 = Deconv2D_Unet('Generator.5.dec', 2*dim, 3, 4, dec5)
    print "5: ", dec5.shape
    output = tf.tanh(dec5)
    return output


def UResNet(samples, dim, dim_out):

    enc = tf.reshape(samples, [-1, 3, 32, 32])
    print "block 1 shape ", enc.shape
    enc_blk = OptimizedResBlockDisc1('Generator.0', enc, dim)
    print "shape of layer {}: {} -> {}".format(0, enc.shape, enc_blk.shape)
    enc1 = UResBlock('Generator.1.enc', dim, dim, 3, enc_blk, resample='down')
    print "shape of enc layer {}: {} -> {}".format(1, enc_blk.shape, enc1.shape)
    enc2 = UResBlock('Generator.2.enc', dim, dim, 3, enc1, resample='down')
    print "shape of enc layer {}: {} -> {}".format(2, enc1.shape, enc2.shape)
    enc3 = UResBlock('Generator.3.enc', dim, dim, 3, enc2, resample='down')
    print "shape of enc layer {}: {} -> {}".format(3, enc2.shape, enc3.shape)
    enc4 = UResBlock('Generator.4.enc', dim, dim, 3, enc3, resample='down')
    print "shape of enc layer {}: {} -> {}".format(4, enc3.shape, enc4.shape)
    print "shape of enc output: {}".format(enc4.shape)
    print
    dec1 = UResBlock('Generator.1.dec', dim, dim, 3, enc4, resample='up', skip=None)
    print "shape of dec layer {}: {} -> {}".format(1, enc4.shape, dec1.shape)
    dec2 = UResBlock('Generator.2.dec', dim, dim, 3, dec1, resample='up', skip=enc3)
    print "shape of dec layer {}: {} -> {}".format(2, dec1.shape, dec2.shape)
    dec3 = UResBlock('Generator.3.dec', dim, dim, 3, dec2, resample='up', skip=enc2)
    print "shape of dec layer {}: {} -> {}".format(3, dec2.shape, dec3.shape)
    dec4 = UResBlock('Generator.4.dec', dim, dim, 3, dec3, resample='up', skip=enc1)
    print "shape of dec layer {}: {} -> {}".format(4, dec3.shape, dec4.shape)
    dec5 = UResBlock('Generator.5.dec', dim, dim, 3, dec4, resample='up', skip=enc_blk, no_norm=True)
    print "shape of dec layer {}: {} -> {}".format(5, dec4.shape, dec5.shape)
    dec = Normalize('Generator.OutputN.dec', dec5)
    dec = prelu(dec, 'dec_prelu')
    dec = Conv2D('Generator.Output.dec', dim, 3, 3, dec, he_init=False)
    print "dec conv output shape: ", dec.shape
    output = tf.tanh(dec)
    return tf.reshape(output, [-1, dim_out])


def GMnist(n_samples, dim, dim_out, noise=None):

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*4*dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*dim, 4, 4])

    output = Deconv2D('Generator.2', 4*dim, 2*dim, 5, output)
    output = tf.nn.relu(output)

    output = output[:, :, :7, :7]

    output = Deconv2D('Generator.3', 2*dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = Deconv2D('Generator.5', dim, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, dim_out])


