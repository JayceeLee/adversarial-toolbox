import os
import numpy as np
import tensorflow as tf
import resnet_v2

slim = tf.contrib.slim
ckpt_dir = '../checkpoints/'
CKPT_LOADED = False


class ResNetV2Model(object):
    def __init__(self, sess):
	self.sess = sess
	self.num_classes = 1001
	self.image_size = 299
	self.num_channels = 3
        self.built = False
        self.ckpt_loaded = CKPT_LOADED
        self.input = tf.placeholder(tf.float32, (None, 299, 299, 3))

    def __call__(self, x_input):

        self._build()
	output = self.end_points['predictions']
	probs = output.op.inputs[0]
	return probs

    def _build(self):
        reuse = True if self.built else None
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(
                    self.input, num_classes=self.num_classes, is_training=False,
                    reuse=reuse)
            self.built = True
        self.end_points = end_points
        self.logits = logits
        if not self.ckpt_loaded:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_dir + 'resnet_v2_101.ckpt')
            self.ckpt_loaded = True

    def _free(self):
        self.sess.close()
        
    def predict(self, x_input):
        if not self.built:
            self._build()
            reuse = True
        logits = self.logits
        output = self.end_points['predictions']
        prob_vals, logit_vals = self.sess.run([output, logits], 
                                         feed_dict={self.input: x_input})
        
        return prob_vals

	

