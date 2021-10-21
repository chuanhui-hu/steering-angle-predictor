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
        self.FC = [512, 512, 64, 32, 1]

        # # parameters to build convolutional layers, length = 5 for 5 layers
        # self.KERNEL_SIZE = [10]
        # self.CHANNEL_NUM = [1]
        # self.STRIDE = [2]
        #
        # # parameters to build fully connected layers, length = 4 for 4 layers
        # self.FC = [1]

        # parameter for the L2 regularization
        self.REGULARIZATION_RATE = 0.001  # 0.0001 the weight of the L2 regularization in the loss function

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

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
                                trainable=True)/10.0
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
                                trainable=True)/10.0
            b = tf.get_variable(name=name + '_b',
                                shape=[output_dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer,
                                trainable=True)

            # W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            # b = tf.Variable(tf.constant(0.1, shape=[output_dim]))

            # add the L2 regularization of the weight to the collection of losses
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)(W))

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

        x_image = x

        # first convolutional layer
        W_conv1 = self.weight_variable([5, 5, 3, 24])
        b_conv1 = self.bias_variable([24])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1, 2) + b_conv1)

        # second convolutional layer
        W_conv2 = self.weight_variable([5, 5, 24, 36])
        b_conv2 = self.bias_variable([36])

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # third convolutional layer
        W_conv3 = self.weight_variable([5, 5, 36, 48])
        b_conv3 = self.bias_variable([48])

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 2) + b_conv3)

        # fourth convolutional layer
        W_conv4 = self.weight_variable([3, 3, 48, 64])
        b_conv4 = self.bias_variable([64])

        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4, 1) + b_conv4)

        # fifth convolutional layer
        W_conv5 = self.weight_variable([3, 3, 64, 64])
        b_conv5 = self.bias_variable([64])

        h_conv5 = tf.nn.relu(self.conv2d(h_conv4, W_conv5, 1) + b_conv5)

        size = h_conv5.shape.as_list()
        h_conv5_flat = tf.reshape(h_conv5, [-1, size[1] * size[2] * size[3]])

        # FCL 1
        W_fc1 = self.weight_variable([size[1] * size[2] * size[3], 1164])
        b_fc1 = self.bias_variable([1164])

        # h_conv5_flat = tf.reshape(h_conv5, [-1, size[1]*size[2]*size[3]])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # FCL 2
        W_fc2 = self.weight_variable([1164, 100])
        b_fc2 = self.bias_variable([100])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # FCL 3
        W_fc3 = self.weight_variable([100, 50])
        b_fc3 = self.bias_variable([50])

        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

        # FCL 3
        W_fc4 = self.weight_variable([50, 10])
        b_fc4 = self.bias_variable([10])

        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

        # Output
        W_fc5 = self.weight_variable([10, 1])
        b_fc5 = self.bias_variable([1])

        # linear
        # y = tf.multiply((tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)
        ## atan
        y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)  # scale the atan output

        # print(y.shape)
        L2NormConst = 0.001

        train_vars = tf.trainable_variables()

        loss = tf.reduce_mean(tf.square(tf.subtract(label_tensor, y))) + \
               tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

        # loss = mse

        return loss, y


def main(*args, **kwargs):
    model = Model()
    model.build_model()


if __name__ == '__main__':
    main()
