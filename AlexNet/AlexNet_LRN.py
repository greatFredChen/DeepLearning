'''
Based on tensorflow 2.1.0
by Fred Chen
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class LRNModel:
    @staticmethod
    def model_init(input_size, model_size, y_size):
        '''
        :param input_size: The shape of data ==> [batch_size, height, width]
        :param model_size: add channel dimension for the data
        ==> [batch_size, height, width, channel]
        :param y_size: The shape of output ==> [batch_size, categories]
        :return:
        '''
        x = tf.compat.v1.placeholder(tf.float32, shape=input_size)
        _x = tf.reshape(x, shape=model_size, name='x')
        y = tf.compat.v1.placeholder(tf.float32, shape=y_size)
        return x, _x, y

    @staticmethod
    def convolution(input, filter, strides, padding='VALID'):
        '''

        :param input: data
        :param filter: parameters of the convolution window
        ==> [height, width, in_channel, out_channel]
        :param strides: strides for width and height ==> [h_stride, w_stride]
        :param padding: The type of padding algorithm to use, including
        SAME: fill the blanks with 0 in the convolution window
        if stride = 1 and width = n ==> output width = n
        VALID: just use the valid data in the convolution window,
        after convolution the width will be smaller than before
        :return:
        '''
        c1 = tf.nn.conv2d(input, filter, strides, padding, name='c1')
        return c1

    @staticmethod
    def local_response_normalization(input, dr, bias, alpha, beta):
        '''

        :param input: data
        :param dr: we will calculate the sum of [a, b, c, d - dr: d + dr + 1]^2
        :param bias: It is k in the LRN expression
        :param alpha: It is alpha in the LRN expression
        :param beta: It is beta in the LRN expression
        expression: b = a / (bias + alpha *
        sum([a, b, c, d - dr: d + dr + 1]^2))^beta
        :return:
        '''
        lrn = tf.nn.local_response_normalization(input, dr,
                                                 bias, alpha, beta,
                                                 name='lrn')
        return lrn

    @staticmethod
    def relu(input):
        '''

        :param input: data
        :return:
        '''
        relu = tf.nn.relu(input, name='relu')
        return relu

    @staticmethod
    def full_connect(input, w, b):
        '''

        :param input: data
        :param w: parameters of neurons
        ==> [input layer neurons, output layer neurons]
        :param b: bias
        ==> [output layer neurons]
        :return:
        '''
        fc = tf.linalg.matmul(input, w) + b  # full_connected = input @ w + b
        return fc

    @staticmethod
    def dropout(input, rate):
        '''

        :param input: data
        :param rate: The probability that each element is dropped
        setting rate=0.1 would drop 10% of input elements randomly.
        :return:
        '''
        dp = tf.nn.dropout(input, rate, name='dropout')
        return dp

    @staticmethod
    def max_pooling(input, ksize, strides, padding='VALID'):
        '''

        :param input: data
        :param ksize: The size of the window for each dimension of the input tensor.
        ==> [height, width]
        :param strides: The stride of the sliding window for each dimension of the
        input tensor.
        ==> [height, width](strides)
        :param padding: SAME or VALID
        :return:
        '''
        mp = tf.nn.max_pool(input, ksize, strides, padding, name='maxpooling')
        return mp

    @staticmethod
    def flatten(input, shape):
        '''

        :param input: data
        :param shape: output shape
        ==> [batch_size, h * w * c]
        :return:
        '''
        # shape = tf.shape(input).numpy()  # (batch_size, h, w, c)
        flatten_input = tf.reshape(input, shape)
        return flatten_input  # (batch_size, h * w * c)


class Parameter:
    @staticmethod
    def kernel(kernel_size, stddev):
        '''
        when generating convolution window, we use normal distribution to
        generate numbers and fill the window with these numbers.
        This method can also generate full-connected layer
        :param kernel_size: convolution window shape
        ==> [height, width, in_channel, out_channel]
        OR full_connected layer size ==> [in_layer, out_layer](neurons)
        :param stddev: standard deviation for the normal distribution
        :return: tf.Variable
        '''
        trunc = tf.random.truncated_normal(kernel_size, stddev=stddev)
        return tf.Variable(trunc)

    @staticmethod
    def bias(channel_size, stddev=1.0):
        '''
        Generate bias for convolution layer or full-connected layer
        :param channel_size: out_channel OR out_layer neurons
        ==>[channel_size](1-D array)
        :return: tf.Variable
        '''
        nr = tf.random.normal(channel_size, stddev=stddev)
        return tf.Variable(nr)

    @staticmethod
    def cost_and_optimizer(y, hx, learning_rate):
        '''

        :param y: initial y
        :param hx: softmax outputs
        :return:
        '''
        loss = tf.compat.v1.reduce_mean(
            -tf.compat.v1.reduce_sum(
                y * tf.compat.v1.log(hx),
                axis=1
            )
        )
        optimizer = tf.compat.v1.\
            train.AdamOptimizer(learning_rate).minimize(loss)
        return loss, optimizer

    @staticmethod
    def pred_and_accuracy(y, hx):
        '''

        :param y: initial y
        :param hx: softmax outputs
        :return:
        '''
        '''
        choose the most probable labels for hx and y.
        Then compare the two labels and give a Tensor output
        which the type of its element is bool.
        '''
        correct = tf.math.equal(tf.math.argmax(y, axis=1),
                                tf.math.argmax(hx, axis=1))
        # cast the correct array to float then calculate the mean
        # accuracy equals to the mean
        accuracy = tf.compat.v1.reduce_mean(
            tf.cast(
                correct,
                dtype=tf.float32
            )
        )
        return accuracy  # 0 to 1
