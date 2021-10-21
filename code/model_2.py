# Author: Chuanhui
# the model.py defines the structure of the training model

import tensorflow as tf


class Model(object):
    def __init__(self):
        # parameters to build convolutional layers, length = 5 for 5 layers
        self.KERNEL_SIZE = [5, 5, 5, 3, 3]
        self.BATCH_NUM = [24, 36, 48, 64, 64]
        self.STRIDE = [2, 2, 2, 1, 1]

        # parameters to build fully connected layers, length = 4 for 4 layers
        self.FC = [1164, 100, 50, 10]

        # parameter for the L2 regularization
        self.REGULARIZATION_RATE = 0.001  # the weight of the L2 regularization in the loss function

    def weight_variable(self, shape, name):
        '''
        initialize the weight
        :param shape: [5, 5, 3, 24]: kernel size: 5*5, input batch: 3, output batch: 24
        :return: tf.Variable(initial_weight)
        '''
        initial_weight = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_weight)

    def bias_variable(self, shape, name):
        '''
        initialize the bias
        :param shape: [24]: output batch: 24
        :return:tf.Variable(initial_bias)
        '''
        initial_bias = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_bias)

    def conv2d(self, x, W, STRIDE):
        '''
        build a 2d convolutional layer
        :param x: input images
        :param W: weight
        :param STRIDE: usually 1 or 2
        :return: tf.nn.conv2d()
        '''
        return tf.nn.conv2d(x, W, strides=[1, STRIDE, STRIDE, 1], padding='VALID')

    def build_model(self, input_tensor, label_tensor, keep_prob):
        '''
        build the CNN model with given coefficients
        :return: None
        '''
        # batch normalization
        x_image = tf.layers.batch_normalization(input_tensor, training=True)

        # build the convolutional layers
        for i in range(len(self.KERNEL_SIZE)):
            if i == 0:  # the first convolutional layer
                W_conv = self.weight_variable([self.KERNEL_SIZE[i], self.KERNEL_SIZE[i], 3, self.BATCH_NUM[i]])
                b_conv = self.bias_variable([self.BATCH_NUM[i]])

                h_conv = tf.nn.elu(self.conv2d(x_image, W_conv, self.STRIDE[i]) + b_conv)
            else:
                W_conv = self.weight_variable(
                    [self.KERNEL_SIZE[i], self.KERNEL_SIZE[i], self.BATCH_NUM[i-1], self.BATCH_NUM[i]])
                b_conv = self.bias_variable([self.BATCH_NUM[i]])

                h_conv = tf.nn.elu(self.conv2d(h_conv, W_conv, self.STRIDE[i]) + b_conv)

        size = h_conv.shape.as_list()  # the output size of the final convolutional layer
        # print(size)
        h_conv_flat = tf.reshape(h_conv, [-1, size[1]*size[2]*size[3]])  # flatten the output for fully connected layers

        # build the fully connected layers
        for i in range(len(self.FC)):
            if i == 0:  # the first fully connected layer
                W_fc = self.weight_variable([size[1]*size[2]*size[3], self.FC[i]])
                b_fc = self.bias_variable([self.FC[i]])

                # add the L2 regularization of the weight to the collection of losses
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(W_fc))

                h_fc = tf.nn.elu(tf.matmul(h_conv_flat, W_fc) + b_fc)

                h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
            else:
                W_fc = self.weight_variable([self.FC[i-1], self.FC[i]])
                b_fc = self.bias_variable([self.FC[i]])

                # add the L2 regularization of the weight to the collection of losses
                tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(W_fc))

                h_fc = tf.nn.elu(tf.matmul(h_fc_drop, W_fc) + b_fc)

                h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
                # print(h_fc_drop.shape.as_list())

        # Output
        W_fc = self.weight_variable([self.FC[-1], 1])
        b_fc = self.bias_variable([1])

        # linear
        # y = tf.multiply((tf.matmul(h_fc_drop, W_fc5) + b_fc5), 3)
        # atan
        y = tf.multiply(tf.atan(tf.matmul(h_fc_drop, W_fc) + b_fc), 3)  # scale the atan output
        # print(y.shape)

        mse = tf.reduce_mean(tf.square(tf.subtract(label_tensor, y)))
        tf.add_to_collection('losses', mse)

        loss = tf.add_n(tf.get_collection('losses'))

        return loss


def main(*args, **kwargs):
    model = Model()
    model.build_model()


if __name__ == '__main__':
    main()