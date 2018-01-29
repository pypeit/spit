""" SPIT Classifier object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
from pkg_resources import resource_filename
import tensorflow as tf
import prettytensor as pt

class Classifier(object):
    """ Class to hold a Tensorflow architecture
    """

    def __init__(self, croot=None, **kwargs):
        """
        Parameters
        ----------
        croot : str, optional
          Path + root of the classifier files
          Currently defaults to kast_original
        kwargs
        """
        # Init
        if croot is None:
            from pkg_resources import resource_filename
            kast_dir = resource_filename('spit', '/data/checkpoints/kast_original/')
            if not os.path.isdir(kast_dir):
                raise IOError("kast_dir {:s} does not exist!".format(kast_dir))
            croot = os.path.join(kast_dir, 'best_validation')
        else:
            import glob
            # Test
            files = glob.glob(croot+'*')
            if len(files) == 0:
                raise IOError("Bad croot to Classifier!")
        # Setup Tensorflow
        self.init_session()
        self.init_variables()
        # Load
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=croot)

    def init_session(self):
        """ Initialize a Tensorflow session
        """
        from spit import image_loader as il
        self.x = tf.placeholder(tf.float32, shape=[None, il.img_size_flat], name='x')
        x_image = tf.reshape(self.x, [-1, il.image_height, il.image_width, il.num_channels])
        self.y_true = tf.placeholder(tf.float32, shape=[None, il.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1)
        x_pretty = pt.wrap(x_image)

        with tf.Graph().as_default(), pt.defaults_scope(activation_fn=tf.nn.relu):
            self.y_pred, loss = x_pretty. \
                conv2d(kernel=5, depth=36, name='layer_conv1'). \
                max_pool(kernel=2, stride=2). \
                conv2d(kernel=5, depth=64, name='layer_conv2'). \
                max_pool(kernel=2, stride=2). \
                flatten(). \
                fully_connected(size=128, name='layer_fc1'). \
                softmax_classifier(num_classes=il.num_classes, labels=self.y_true)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.session = tf.Session()
        # Return
        return

    def init_variables(self):
        """ Initialize variables
        """
        self.session.run(tf.global_variables_initializer())
        return
