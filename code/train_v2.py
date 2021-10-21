# Author: Chuanhui
# The train.py runs the training step of the model
# The model is a CNN model

import scipy.misc
import model_3
import data_processing
import datetime
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class TrainProcess(model_3.Model):
    def __init__(self):
        '''
        BATCH_SIZE: the batch size of every iteration
        LEARNING_RATE_BASE: the initial learning rate
        LEARNING_RATE_DECAY: the dacay rate of the learning rate
        TRAINING_EPOCH: the whole training epochs
        LOGDIR: the directory to save the model
        '''
        model_3.Model.__init__(self)
        self.BATCH_SIZE = 64
        self.LEARNING_RATE_BASE = 0.01
        self.LEARNING_RATE_DECAY = 0.99
        self.TRAINING_EPOCH = 30

        self.LOGDIR = './model_' + str(datetime.datetime.now())[:10] + '/'

    def save_model(self, saver, sess, epoch_num):
        '''
        save the model to certain directory
        :param sess: the model to save
        :param epoch_num: current number of epochs
        :return: None
        '''
        if not os.path.exists(self.LOGDIR + '%i/' % epoch_num):
            os.mkdir(self.LOGDIR + '%i/' % epoch_num)
        checkpoint_path = os.path.join(self.LOGDIR + '%i/' % epoch_num, "model.ckpt")
        filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)

    def train(self):
        '''
        build the CNN model with given coefficients and train the model
        :return: None
        '''
        if not os.path.exists(self.LOGDIR):
            os.mkdir(self.LOGDIR)
        with open(self.LOGDIR + 'model_info.txt', 'a+') as infile:
            infile.write('KERNEL_SIZE = ' + str(self.KERNEL_SIZE) + '\n')
            infile.write('CHANNEL_NUM = ' + str(self.CHANNEL_NUM) + '\n')
            infile.write('STRIDE = ' + str(self.STRIDE) + '\n')
            infile.write('FC = ' + str(self.FC) + '\n')
            infile.write('REGULARIZATION_RATE = ' + str(self.REGULARIZATION_RATE) + '\n')
            infile.write('BATCH_SIZE = ' + str(self.BATCH_SIZE) + '\n')
            infile.write('LEARNING_RATE_BASE = ' + str(self.LEARNING_RATE_BASE) + '\n')
            infile.write('LEARNING_RATE_DECAY = ' + str(self.LEARNING_RATE_DECAY) + '\n')
            infile.write('TRAINING_EPOCH = ' + str(self.TRAINING_EPOCH) + '\n')

        # x = tf.placeholder(tf.float32, shape=[None, 160, 320, 3])  # input images, [BATCH_NUM, height, width, channels]
        x = tf.placeholder(tf.float32, shape=[None, 120, 320, 3])  # input images, [BATCH_NUM, height, width, channels]

        y_ = tf.placeholder(tf.float32, shape=[None, 1])  # label(ground truth), [BATCH_NUM, output_num]

        train_data = data_processing.Data(flag='train')
        validation_data = data_processing.Data(flag='validation')

        keep_prob = tf.placeholder(tf.float32)  # the rate of dropout

        # loss = self.build_model(input_tensor=x, label_tensor=y_, keep_prob=keep_prob)  # the loss to train on

        global_step = tf.Variable(0, trainable=False)  # the global_step for gradient decay, trainable=False

        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE_BASE,  # the initial learning rate
                                                   global_step,  # current step
                                                   train_data.data_num/self.BATCH_SIZE,  # num of steps to take for 1 epoch
                                                   self.LEARNING_RATE_DECAY)  # the decay rate of learning rate

        # x = input_tensor

        # build the convolutional layers
        # conv0
        x0 = tf.layers.batch_normalization(x, training=True)
        x0_out = self.conv_layer(x=x0,
                                 kernel_size=self.KERNEL_SIZE[0],
                                 output_channel=self.CHANNEL_NUM[0],
                                 stride=self.STRIDE[0],
                                 name='conv_layer_%i' % 0)

        # conv1
        # x1 = tf.layers.batch_normalization(x0_out, training=True)
        # x1_out = self.conv_layer(x=x1,
        #                          kernel_size=self.KERNEL_SIZE[1],
        #                          output_channel=self.CHANNEL_NUM[1],
        #                          stride=self.STRIDE[1],
        #                          name='conv_layer_%i' % 1)
        x1_out = self.conv_layer(x=x0_out,
                                 kernel_size=self.KERNEL_SIZE[1],
                                 output_channel=self.CHANNEL_NUM[1],
                                 stride=self.STRIDE[1],
                                 name='conv_layer_%i' % 1)

        # conv2
        # x2 = tf.layers.batch_normalization(x1_out, training=True)
        # x2_out = self.conv_layer(x=x2,
        #                          kernel_size=self.KERNEL_SIZE[2],
        #                          output_channel=self.CHANNEL_NUM[2],
        #                          stride=self.STRIDE[2],
        #                          name='conv_layer_%i' % 2)
        x2_out = self.conv_layer(x=x1_out,
                                 kernel_size=self.KERNEL_SIZE[2],
                                 output_channel=self.CHANNEL_NUM[2],
                                 stride=self.STRIDE[2],
                                 name='conv_layer_%i' % 2)

        # conv3
        # x3 = tf.layers.batch_normalization(x2_out, training=True)
        # x3_out = self.conv_layer(x=x3,
        #                          kernel_size=self.KERNEL_SIZE[3],
        #                          output_channel=self.CHANNEL_NUM[3],
        #                          stride=self.STRIDE[3],
        #                          name='conv_layer_%i' % 3)
        x3_out = self.conv_layer(x=x2_out,
                                 kernel_size=self.KERNEL_SIZE[3],
                                 output_channel=self.CHANNEL_NUM[3],
                                 stride=self.STRIDE[3],
                                 name='conv_layer_%i' % 3)

        # conv4
        # x4 = tf.layers.batch_normalization(x3_out, training=True)
        # x4_out = self.conv_layer(x=x4,
        #                          kernel_size=self.KERNEL_SIZE[4],
        #                          output_channel=self.CHANNEL_NUM[4],
        #                          stride=self.STRIDE[4],
        #                          name='conv_layer_%i' % 4)
        x4_out = self.conv_layer(x=x3_out,
                                 kernel_size=self.KERNEL_SIZE[4],
                                 output_channel=self.CHANNEL_NUM[4],
                                 stride=self.STRIDE[4],
                                 name='conv_layer_%i' % 4)

        size = x4_out.shape.as_list()  # the output size of the final convolutional layer
        # print(size)
        x_fc = tf.reshape(x4_out, [-1, size[1] * size[2] * size[3]])  # flatten the output for fully connected layers

        # build the fully connected layers
        x0_fc = self.fully_connect(x=x_fc,
                                   output_dim=self.FC[0],
                                   name='fully_connected_%i' % 0,
                                   keep_prob=keep_prob)

        # build the fully connected layers
        x1_fc = self.fully_connect(x=x0_fc,
                                   output_dim=self.FC[1],
                                   name='fully_connected_%i' % 1,
                                   keep_prob=keep_prob)

        # build the fully connected layers
        x2_fc = self.fully_connect(x=x1_fc,
                                   output_dim=self.FC[2],
                                   name='fully_connected_%i' % 2,
                                   keep_prob=keep_prob)

        # build the fully connected layers
        x3_fc = self.fully_connect(x=x2_fc,
                                   output_dim=self.FC[3],
                                   name='fully_connected_%i' % 3,
                                   keep_prob=keep_prob)

        # build the fully connected layers
        x4_fc = self.fully_connect(x=x3_fc,
                                   output_dim=self.FC[4],
                                   name='fully_connected_%i' % 4,
                                   keep_prob=keep_prob)


        # linear
        y = tf.multiply(x4_fc, 1)
        # atan
        # y = tf.multiply(tf.atan(x4_fc), 10)  # scale the atan output
        # print(y.shape)

        mse = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
        tf.add_to_collection('losses', mse)

        loss = tf.add_n(tf.get_collection('losses'))
        # loss = mse

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        # op to write logs to Tensorboard
        logs_path = './logs'
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # print all trainable variables
        variable_names = [v.name for v in tf.trainable_variables()]
        print(variable_names)

        #################################################################
        #                the training process starts here               #
        #################################################################

        for epoch in range(self.TRAINING_EPOCH):  # total training epochs = self.TRAINING_EPOCH
            # train_data.shuffle_data()  # shuffle the data every epoch

            if epoch == 0:
                train_data.shuffle_data()  # shuffle the data every epoch

            iter_per_epoch = int(np.ceil(train_data.data_num/self.BATCH_SIZE))  # iterations per epoch

            for i in range(iter_per_epoch):  # read training images and labels for input_x
                if i < iter_per_epoch-1:  # full batch size
                    batch_image = train_data.shuffled_list[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
                else:  # the last piece of the training set
                    batch_image = train_data.shuffled_list[i*self.BATCH_SIZE:-1]
                # input_x = np.zeros([len(batch_image), 160, 320, 3], dtype=np.float32)
                input_full = np.zeros([len(batch_image), 160, 320, 3], dtype=np.float32)
                label = np.zeros([len(batch_image), 1], dtype=np.float32)
                for j in range(len(batch_image)):
                    input_full[j, :, :, :] = plt.imread(batch_image[j])  # read the input batch into an 4D array
                    label[j] = train_data.steering_dict[batch_image[j][16:]]*scipy.pi/1800.0  # match the images with labels
                    # print(batch_image[j] + ', ' + batch_image[j][16:], label[j])
                input_x = input_full[:, 40:, :, :]
                # training step
                # train_step.run(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 0.8})

                #####################################################
                # weight_var = tf.get_collection(tf.GraphKeys.VARIABLES, "conv_layer_4/conv_layer_4_W:0")[0]
                # print('weight: ', sess.run(tf.get_default_graph().get_tensor_by_name('conv_layer_4/conv_layer_4_W:0')))
                # print('output of conv layer 3: ', sess.run(x3_out, feed_dict={x: input_x, y_: label, keep_prob: 1.0}))
                # print('output of conv layer 4: ', sess.run(x4_out, feed_dict={x: input_x, y_: label, keep_prob: 1.0}))
                # print('output of the first FC layer: ', sess.run(x0_fc, feed_dict={x: input_x, y_: label, keep_prob: 1.0}))
                #####################################################

                # evaluate the loss on training set and log
                if np.mod(i, 10) == 0:
                    loss_value = loss.eval(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 1.0})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * self.BATCH_SIZE + i, loss_value))

                    if not os.path.exists(self.LOGDIR):
                        os.mkdir(self.LOGDIR)
                    with open(self.LOGDIR + 'training_log.txt', 'a+') as infile:
                        infile.write("Epoch: %d, Step: %d, Loss: %g \n" % (epoch, epoch * self.BATCH_SIZE + i, loss_value))

                # training step
                train_step.run(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 0.8})

                # write logs at every iteration
                summary = merged_summary_op.eval(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 1.0})
                summary_writer.add_summary(summary, epoch * self.BATCH_SIZE + i)

            #################################################################
            #                 the training process ends here                #
            #################################################################

            # evaluate the loss on validation set and log
            loss_value_val = 0
            val_image = validation_data.list_of_image
            input_val_full = np.zeros([1, 160, 320, 3], dtype=np.float32)
            # print(input_val.shape)
            label_val = np.zeros([1, 1], dtype=np.float32)
            for i in range(validation_data.data_num):
                img = plt.imread(val_image[i])
                input_val_full[0, :, :, :] = img  # read the validation image into an 4D array
                input_val = input_val_full[:, 40:, :, :]
                # print(val_image[i][20:])
                label_val[0, :] = validation_data.steering_dict[val_image[i][21:]]*scipy.pi/1800.0
                loss_value_val += loss.eval(session=sess, feed_dict={x: input_val, y_: label_val, keep_prob: 1.0})
            print("Epoch: %d, Loss: %g" % (epoch, loss_value_val/validation_data.data_num))

            with open(self.LOGDIR + 'validation_log.txt', 'a+') as infile:
                infile.write("Epoch: %d, Loss: %g \n" % (epoch, loss_value_val/validation_data.data_num))

            # save the model every 10 epochs
            if np.mod(epoch, 10) == 0:
                self.save_model(saver=saver, sess=sess, epoch_num=epoch)

        sess.close()

        print("Run the command line:\n"
              "--> tensorboard --logdir=./logs "
              "\nThen open http://0.0.0.0:6006/ into your web browser")


def main(*args, **kwargs):
    train = TrainProcess()
    train.train()


if __name__ == '__main__':
    main()
