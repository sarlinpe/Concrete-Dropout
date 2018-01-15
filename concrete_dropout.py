import tensorflow as tf
import numpy as np

from tensorflow.python.layers import base
from tensorflow.python.layers import utils

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops


class ConcreteDropout(base.Layer):
    """Concrete Dropout layer class from https://arxiv.org/abs/1705.07832.

    "Concrete Dropout" Yarin Gal, Jiri Hron, Alex Kendall

    Arguments:
        weight_regularizer:
            Positive float, satisfying $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
            (inverse observation noise), and N the number of instances
            in the dataset.
        dropout_regularizer:
            Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min:
            Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min:
            Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name:
            String, name of the layer.
        reuse:
            Boolean, whether to reuse the weights of a previous layer
            by the same name.
    """

    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, name=None, reuse=False,
                 training=True, **kwargs):

        super(ConcreteDropout, self).__init__(name=name, _reuse=reuse,
                                              **kwargs)
        assert init_min <= init_max, \
            'init_min must be lower or equal to init_max.'

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))
        self.training = training
        self.reuse = reuse

    def get_kernel_regularizer(self):
        def kernel_regularizer(weight):
            if self.reuse:
                return None
            return self.weight_regularizer * tf.reduce_sum(tf.square(weight)) \
                / (1. - self.p)
        return kernel_regularizer

    def apply_dropout_regularizer(self, inputs):
        with tf.name_scope('dropout_regularizer'):
            input_dim = tf.cast(tf.reduce_prod(tf.shape(inputs)[1:]),
                                dtype=tf.float32)
            dropout_regularizer = self.p * tf.log(self.p)
            dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 dropout_regularizer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.input_spec = base.InputSpec(shape=input_shape)

        self.p_logit = self.add_variable(name='p_logit',
                                         shape=[],
                                         initializer=tf.random_uniform_initializer(
                                             self.init_min,
                                             self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit, name='dropout_rate')
        tf.add_to_collection('DROPOUT_RATES', self.p)

        self.built = True

    def concrete_dropout(self, x):
        eps = 1e-7
        temp = 0.1

        with tf.name_scope('dropout_mask'):
            unif_noise = tf.random_uniform(shape=tf.shape(x))
            drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
            )
            drop_prob = tf.nn.sigmoid(drop_prob / temp)

        with tf.name_scope('drop'):
            random_tensor = 1. - drop_prob
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob

        return x

    def call(self, inputs, training=True):
        def dropped_inputs():
            return self.concrete_dropout(inputs)
        if not self.reuse:
            self.apply_dropout_regularizer(inputs)
        return utils.smart_cond(training,
                                dropped_inputs,
                                lambda: array_ops.identity(inputs))


def concrete_dropout(inputs,
                     trainable=True,
                     weight_regularizer=1e-6,
                     dropout_regularizer=1e-5,
                     init_min=0.1, init_max=0.1,
                     training=True,
                     name=None,
                     reuse=False,
                     **kwargs):

    """Functional interface for Concrete Dropout.

    "Concrete Dropout" Yarin Gal, Jiri Hron, Alex Kendall
    from https://arxiv.org/abs/1705.07832.

    Arguments:
        weight_regularizer:
            Positive float, satisfying $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
            (inverse observation noise), and N the number of instances
            in the dataset.
        dropout_regularizer:
            Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min:
            Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min:
            Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name:
            String, name of the layer.
        reuse:
            Boolean, whether to reuse the weights of a previous layer
            by the same name.

    Returns:
        Tupple containing:
            - the output of the dropout layer;
            - the kernel regularizer function for the subsequent
              convolutional layer.
    """

    layer = ConcreteDropout(weight_regularizer=weight_regularizer,
                            dropout_regularizer=dropout_regularizer,
                            init_min=init_min, init_max=init_max,
                            training=training,
                            trainable=trainable,
                            name=name,
                            reuse=reuse)
    return layer.apply(inputs, training=training), \
        layer.get_kernel_regularizer()
