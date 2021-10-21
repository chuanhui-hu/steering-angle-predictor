# Author: Chuanhui
# The train.py runs the training step of the model
# The model is a CNN model

from model_NVIDIA import Model
import data_processing
import datetime
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


class TrainProcess(Model):
    def __init__(self):
        '''
        BATCH_SIZE: the batch size of every iteration
        LEARNING_RATE_BASE: the initial learning rate
        LEARNING_RATE_DECAY: the dacay rate of the learning rate
        TRAINING_EPOCH: the whole training epochs
        LOGDIR: the directory to save the model
        '''
        Model.__init__(self)
        self.BATCH_SIZE = 64*5
        self.LEARNING_RATE_BASE = 0.001
        self.LEARNING_RATE_DECAY = 0.99
        self.TRAINING_EPOCH = 300

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
        # input images, [BATCH_NUM, height, width, channels]
        x = tf.placeholder(tf.float32, shape=[None, 80, 160, 3])

        # label(ground truth), [BATCH_NUM, output_num]
        y_ = tf.placeholder(tf.float32, shape=[None, 1])

        train_data = data_processing.Data(flag='train')
        validation_data = data_processing.Data(flag='validation')

        keep_prob = tf.placeholder(tf.float32)  # the rate of dropout

        loss, y = self.build_model(input_tensor=x, label_tensor=y_,
                                   keep_prob=keep_prob)  # the loss to train on

        # the global_step for gradient decay, trainable=False
        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE_BASE,  # the initial learning rate
                                                   global_step,  # current step
                                                   train_data.data_num/self.BATCH_SIZE,  # num of steps to take for 1 epoch
                                                   self.LEARNING_RATE_DECAY)  # the decay rate of learning rate

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(
                learning_rate).minimize(loss, global_step=global_step)

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

        for epoch in range(self.TRAINING_EPOCH + 1):  # total training epochs = self.TRAINING_EPOCH
            train_data.shuffle_data()  # shuffle the data every epoch

            # if epoch == 0:
            #     train_data.shuffle_data()  # shuffle the data every epoch

            iter_per_epoch = int(np.ceil(train_data.data_num/self.BATCH_SIZE)
                                 )  # iterations per epoch

            for i in range(iter_per_epoch):  # read training images and labels for input_x
                if i < iter_per_epoch-1:  # full batch size
                    batch_image = train_data.shuffled_list[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
                else:  # the last piece of the training set
                    batch_image = train_data.shuffled_list[i*self.BATCH_SIZE:-1]
                # input_x = np.zeros([len(batch_image), 160, 320, 3], dtype=np.float32)
                input_full = np.zeros([len(batch_image), 80, 160, 3], dtype=np.float32)
                label = np.zeros([len(batch_image), 1], dtype=np.float32)
                for j in range(len(batch_image)):
                    input_full[j, :, :, :] = train_data.resize_image(plt.imread(
                        batch_image[j]))  # read the input batch into an 4D array
                    label[j] = train_data.steering_dict[batch_image[j][16:]] * scipy.pi / \
                        1800.0  # * scipy.pi/1800.0  # match the images with labels
                    # print(batch_image[j] + ', ' + batch_image[j][16:], label[j])
                input_x = input_full[:, :, :, :]
                # training step
                # train_step.run(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 0.8})

                # evaluate the loss on training set and log
                if np.mod(i, 10) == 0:
                    loss_value = loss.eval(session=sess, feed_dict={
                                           x: input_x, y_: label, keep_prob: 1.0})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, np.floor(
                        epoch * train_data.data_num / self.BATCH_SIZE) + i, loss_value))

                    if not os.path.exists(self.LOGDIR):
                        os.mkdir(self.LOGDIR)
                    with open(self.LOGDIR + 'training_log.txt', 'a+') as infile:
                        infile.write("Epoch: %d, Step: %d, Loss: %g \n" % (epoch, np.floor(
                            epoch * train_data.data_num / self.BATCH_SIZE) + i, loss_value))

                # training step
                train_step.run(session=sess, feed_dict={x: input_x, y_: label, keep_prob: 0.8})

                # write logs at every iteration
                summary = merged_summary_op.eval(session=sess, feed_dict={
                                                 x: input_x, y_: label, keep_prob: 1.0})
                summary_writer.add_summary(summary, epoch * self.BATCH_SIZE + i)

            #################################################################
            #                 the training process ends here                #
            #################################################################

            # evaluate the loss on validation set and log
            loss_value_val = 0
            val_image = validation_data.list_of_image
            input_val_full = np.zeros([1, 80, 160, 3], dtype=np.float32)
            # print(input_val.shape)
            label_val = np.zeros([1, 1], dtype=np.float32)
            for i in range(validation_data.data_num):
                img = plt.imread(val_image[i])
                input_val_full[0, :, :, :] = validation_data.resize_image(
                    img)  # read the validation image into an 4D array
                input_val = input_val_full[:, :, :, :]
                # print(val_image[i][20:])
                label_val[0, :] = validation_data.steering_dict[val_image[i]
                                                                [21:]] * scipy.pi/1800.0  # * scipy.pi/1800.0
                loss_value_val += loss.eval(session=sess,
                                            feed_dict={x: input_val, y_: label_val, keep_prob: 1.0})
            print("Epoch: %d, Loss: %g" % (epoch, loss_value_val/validation_data.data_num))

            with open(self.LOGDIR + 'validation_log.txt', 'a+') as infile:
                infile.write("Epoch: %d, Loss: %g \n" %
                             (epoch, loss_value_val/validation_data.data_num))

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
