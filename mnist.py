from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import layers as tfl

from tensorflow.examples.tutorials.mnist import input_data
from concrete_dropout import concrete_dropout


def net(inputs, is_training):

    x = tf.reshape(inputs, [-1, 28, 28, 1])

    dropout_params = {'init_min': 0.1, 'init_max': 0.1,
                      'weight_regularizer': 1e-6, 'dropout_regularizer': 1e-5,
                      'training': is_training}
    x, reg = concrete_dropout(x, name='conv1_dropout', **dropout_params)
    x = tfl.conv2d(x, 32, 5, activation=tf.nn.relu, padding='SAME',
                   kernel_regularizer=reg, bias_regularizer=reg,
                   name='conv1')
    x = tfl.max_pooling2d(x, 2, 2, padding='SAME', name='pool1')

    x, reg = concrete_dropout(x, name='conv2_dropout', **dropout_params)
    x = tfl.conv2d(x, 64, 5, activation=tf.nn.relu, padding='SAME',
                   kernel_regularizer=reg, bias_regularizer=reg,
                   name='conv2')
    x = tfl.max_pooling2d(x, 2, 2, padding='SAME', name='pool2')

    x = tf.reshape(x, [-1, 7*7*64], name='flatten')
    x, reg = concrete_dropout(x, name='fc1_dropout', **dropout_params)
    x = tfl.dense(x, 1024, activation=tf.nn.relu, name='fc1',
                  kernel_regularizer=reg, bias_regularizer=reg)

    outputs = tfl.dense(x, 10, name='fc2')
    return outputs


def main(_):
    mnist = input_data.read_data_sets('MNIST_data')

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    y_out = net(x, is_training)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=y, logits=y_out))
        loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    dropout_rates = tf.get_collection('DROPOUT_RATES')
    def rates_pretty_print(values):
        return {str(t.name): round(r, 4)
                for t, r in zip(dropout_rates, values)}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(40000):
            batch = mnist.train.next_batch(50)
            if i % 500 == 0:
                training_loss, training_acc, rates = sess.run(
                        [loss, accuracy, dropout_rates],
                        feed_dict={
                            x: batch[0], y: batch[1], is_training: False})
                print('step {}, loss {}, accuracy {}'.format(
                    i, training_loss, training_acc))
                print('dropout rates: {}'.format(rates_pretty_print(rates)))
            train_step.run(feed_dict={
                x: batch[0], y: batch[1], is_training: True})

        accuracy, rates = sess.run([accuracy, dropout_rates],
                                   feed_dict={x: mnist.test.images,
                                              y: mnist.test.labels,
                                              is_training: False})
        print('test accuracy {}'.format(accuracy))
        print('final dropout rates: {}'.format(rates_pretty_print(rates)))


if __name__ == '__main__':
    tf.app.run(main=main)
