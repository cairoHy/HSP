import os
import tensorflow as tf


class Model:
    """ Abstract object representing a main model """

    def __init__(self, params, scope):
        self.scope = scope
        self._params = params
        self.build_graph()
        self.saver = tf.train.Saver(tf.trainable_variables())

    @staticmethod
    def save(saver, sess, model_dir, model_prefix):
        """Saves the model into model_dir with model_prefix as the model indicator"""
        saver.save(sess, os.path.join(model_dir, model_prefix))
        tf.logging.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, sess, model_dir):
        """Restores the model into model_dir from model_prefix as the model indicator"""
        model_file = tf.train.latest_checkpoint(model_dir)
        self.saver.restore(sess, model_file)
        tf.logging.info('Model restored from {}'.format(model_file))

    def build_graph(self):
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params
