# Author: Chuanhui
# the run_model.py runs the trained model on the testset

import model_NVIDIA
import data_processing
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy


class RunModel(model_NVIDIA.Model):
    def __init__(self):
        '''
        BATCH_SIZE: the batch size of every iteration
        LEARNING_RATE_BASE: the initial learning rate
        LEARNING_RATE_DECAY: the dacay rate of the learning rate
        TRAINING_EPOCH: the whole training epochs
        LOGDIR: the directory to save the model
        loss: the loss value of each test image
        '''
        model_NVIDIA.Model.__init__(self)
        self.BATCH_SIZE = 64
        self.LEARNING_RATE_BASE = 0.01
        self.LEARNING_RATE_DECAY = 0.99

        self.LOGDIR = './model_2019-04-23/'

        self.loss = []

    def restore_model(self, saver, sess, epoch_num):
        '''
        restore the model from certain directory
        :param sess: the model to restore
        :param epoch_num: the model saved at which epoch number
        :return: None
        '''
        checkpoint_path = os.path.join(self.LOGDIR + '%i/' % epoch_num, "model.ckpt")
        saver.restore(sess, checkpoint_path)
        # print("Model restored from file: %s" % checkpoint_path)

    def build_sess(self):
        '''
        rebuild the model and the sess
        :return: None
        '''
        self.x = tf.placeholder(tf.float32, shape=[None, 80, 160, 3])  # input images, [BATCH_NUM, height, width, channels]

        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])  # label(ground truth), [BATCH_NUM, output_num]


        self.keep_prob = tf.placeholder(tf.float32)  # the rate of dropout

        self.loss, self.y = self.build_model(input_tensor=self.x,
                                             label_tensor=self.y_,
                                             keep_prob=self.keep_prob)  # the loss to train on

        self.sess = tf.Session()

    def predict_whole(self):
        '''
        restore the trained model and make prediction with the model
        :return: list of loss values on the whole test set
        '''
        # x = tf.placeholder(tf.float32, shape=[None, 160, 320, 3])  # input images, [BATCH_NUM, height, width, channels]

        test_data = data_processing.Data(flag='test')

        self.build_sess()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        self.restore_model(sess=self.sess, saver=saver, epoch_num=300)

        # evaluate the loss on validation set and log
        loss_value_test = []
        y_value_test = []
        label_value_test = []
        test_image = test_data.list_of_image
        input_test_full = np.zeros([1, 80, 160, 3], dtype=np.float32)
        # print(input_val.shape)
        label_test = np.zeros([1, 1], dtype=np.float32)
        for i in range(test_data.data_num):
            print(i)
            img = plt.imread(test_image[i])
            input_test_full[0, :, :, :] = test_data.resize_image(img)  # read the validation image into an 4D array
            input_test = input_test_full[:, :, :, :]
            # print(val_image[i][20:])
            label_test[0, :] = test_data.steering_dict[test_image[i][15:]] /10
            # print(test_image[i][14:])

            loss_value_test.append((self.y.eval(session=self.sess, feed_dict={self.x: input_test,
                                                                                self.y_: label_test,
                                                                                self.keep_prob: 1.0})/scipy.pi*180 - label_test[0, 0])[0])
            # y_value_test.append(self.y.eval(session=self.sess, feed_dict={self.x: input_test,
            #                                                                     self.y_: label_test,
            #                                                                     self.keep_prob: 1.0}))
            # label_value_test.append(test_data.steering_dict[test_image[i][15:]] / 10)
            # loss_value_test.append(y.eval(session=sess, feed_dict={x: input_test, y_: label_test, keep_prob: 1.0})[0])
        print('average loss is: ', sum(loss_value_test)/len(loss_value_test))

        self.sess.close()
        return loss_value_test

    def predict_one_image(self, img):
        # self.build_sess()
        test_data = data_processing.Data(flag='test')

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.restore_model(sess=self.sess, saver=saver, epoch_num=300)
        input_test_full = np.zeros([1, 80, 160, 3], dtype=np.float32)
        input_test_full[0, :, :, :] = test_data.resize_image(img)
        input_test = input_test_full[:, :, :, :]

        angle = self.y.eval(session=self.sess, feed_dict={self.x: input_test, self.keep_prob: 1.0}) /scipy.pi*180
        # self.sess.close()
        return angle


def main(*args, **kwargs):
    test = RunModel()
    loss, y, label = test.predict_whole()
    print(loss)
    plt.hist(np.sign(loss)*np.sqrt(np.abs(loss)), bins=list(range(-30, 31)))
    plt.show()
    # prediction = test.predict()
    # print(prediction)
    # plt.hist(prediction, bins=list(range(-30, 31)))
    # plt.show()


if __name__ == '__main__':
    main()
