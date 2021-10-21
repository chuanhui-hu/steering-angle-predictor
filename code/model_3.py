# Author: Chuanhui
# the model.py defines the structure of the training model

import tensorflow as tf


class Model(object):
    def __init__(self):
        '''
        KERNEL_SIZE: the kernel size of each convolutional layers
        CHANNEL_NUM: the number of the output channels of each convolutional layer
        STRIDE: the stride of each convolutional layer, can be 1 or 2
        FC: the output dimension of each fully-connected layer, the last one has to be 1
        REGULARIZATION_RATE: the weight of the L2 regularization in the loss function
        '''
        # parameters to build convolutional layers, length = 5 for 5 layers
        self.KERNEL_SIZE = [5, 5, 5, 3, 3]
        self.CHANNEL_NUM = [24, 36, 48, 64, 64]
        self.STRIDE = [2, 2, 1, 1, 1]

        # parameters to build fully connected layers, length = 4 for 4 layers
        self.FC = [512, 512, 128, 32, 1]

        # # parameters to build convolutional layers, length = 5 for 5 layers
        # self.KERNEL_SIZE = [10]
        # self.CHANNEL_NUM = [1]
        # self.STRIDE = [2]
        #
        # # parameters to build fully connected layers, length = 4 for 4 layers
        # self.FC = [1]

        # parameter for the L2 regularization
        self.REGULARIZATION_RATE = 0.001  # 0.0001 the weight of the L2 regularization in the loss function

    def conv_layer(self, x, kernel_size, output_channel, stride, name):
        '''
        construct a convolutional layer
        :param x: input tensor
        :param kernel_size: the size of the kernel
        :param output_channel: output channel
        :param name: the variable scope
        :return: output tensor
        '''
        with tf.variable_scope(name):
            input_channel = x.shape[-1]
            # input_channel = 3
            W = tf.get_variable(name=name + '_W',
                                shape=[kernel_size, kernel_size, input_channel, output_channel],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer,
                                trainable=True)/20.0
            b = tf.get_variable(name=name + '_b',
                                shape=[output_channel],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer,
                                trainable=True)
            # W = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, input_channel, output_channel], stddev=0.1))
            # b = tf.Variable(tf.constant(0.1, shape=[output_channel]))

            return self.leaky_relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')+b, k1=1, k2=0.1)

    def fully_connect(self, x, output_dim, name, keep_prob):
        '''
        construct a fully-connected layer
        :param x: input tensor
        :param output_dim: the output dimension
        :param name: the variable scope
        :return: output tensor
        '''
        with tf.variable_scope(name):
            input_dim = x.shape[-1]
            # input_dim = 8736
            W = tf.get_variable(name=name + '_W',
                                shape=[input_dim, output_dim],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer,
                                trainable=True)/20.0
            b = tf.get_variable(name=name + '_b',
                                shape=[output_dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer,
                                trainable=True)

            # W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            # b = tf.Variable(tf.constant(0.1, shape=[output_dim]))

            # add the L2 regularization of the weight to the collection of losses
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(
                self.REGULARIZATION_RATE)(W))

            return tf.nn.dropout(self.leaky_relu(tf.matmul(x, W) + b, k1=1, k2=0.1), keep_prob=keep_prob)

    def leaky_relu(self, input_tensor, k1=1, k2=0.1):
        '''
        :param input_tensor: the input tensor
        :param k1: the slope for x >= 0
        :param k2: the slope for x < 0
        :return: the output tensor of the activation function
        '''
        return tf.maximum(k1 * input_tensor, k2 * input_tensor)

    def build_model(self, input_tensor, label_tensor, keep_prob):
        '''
        build the CNN model with given coefficients
        :return: None
        '''
        # batch normalization
        # x = tf.layers.batch_normalization(input_tensor, training=True)
        x = input_tensor

        # build the convolutional layers
        for i in range(len(self.KERNEL_SIZE)):
            x = tf.layers.batch_normalization(x, training=True)
            x = self.conv_layer(x=x,
                                kernel_size=self.KERNEL_SIZE[i],
                                output_channel=self.CHANNEL_NUM[i],
                                stride=self.STRIDE[i],
                                name='conv_layer_%i' % i)

        size = x.shape.as_list()  # the output size of the final convolutional layer
        print(size)
        # flatten the output for fully connected layers
        x = tf.reshape(x, [-1, size[1]*size[2]*size[3]])

        # build the fully connected layers
        for i in range(len(self.FC)):
            x = self.fully_connect(x=x,
                                   output_dim=self.FC[i],
                                   name='fully_connected_%i' % i,
                                   keep_prob=keep_prob)

        # linear
        # y = tf.multiply(x, 1)
        # atan
        y = tf.multiply(tf.atan(x), 3)  # scale the atan output
        # print(y.shape)

        mse = tf.reduce_mean(tf.square(tf.subtract(label_tensor, y)))
        tf.add_to_collection('losses', mse)

        loss = tf.add_n(tf.get_collection('losses'))
        # loss = mse

        return loss, y


def main(*args, **kwargs):
    model = Model()
    model.build_model()


if __name__ == '__main__':
    main()
