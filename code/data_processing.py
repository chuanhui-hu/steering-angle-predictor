# Author: Chuanhui
# the data_processing.py pre-process the image data and the steering angle data

import glob
import pandas as pd
import random
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np


class Data(object):
    def __init__(self, flag='train'):
        '''
        path: the path of the training/ validation/ testing data
        set_path(flag): set the path of dataset
        list_of_image: the list of all videos or images in the dataset
        shuffled_list: the shuffled list of videos or images
        data_num: the total number of images in the dataset
        steering_dict: the dictionary matches the name of png files and the steering angles
        read_datalist(): read the name of image files and save in a list
        read_info(): read the steering angles and save in a list
        :param flag: can be 'train', 'validation' or 'test'
        '''
        self.path = []  # the path of the training/ validation/ testing data
        self.set_path(flag)  # set the path of dataset
        self.list_of_image = []  # the list of all videos or images in the dataset
        self.shuffled_list = []  # the shuffled list of videos or images
        self.data_num = 0  # the total number of images in the dataset
        self.steering_dict = {}  # the dictionary matches the name of png files and the steering angles
        self.read_datalist()  # read the name of image files and save in a list
        self.read_info()  # read the steering angles and save in a list

    def set_path(self, flag='train'):
        '''
        set the path of data
        :param flag: 'train' or 'validation' or 'test'
        :return: self.path
        '''
        if flag == 'train':
            self.path = './dataset/train/'
        elif flag == 'validation':
            self.path = './dataset/validation/'
        else:  # flag == 'test'
            self.path = './dataset/test/'
        return self.path

    def read_datalist(self):
        '''
        read the list of names of videos or images
        :return: self.list_of_image
        '''
        self.list_of_image = glob.glob(self.path + '*.png')
        self.data_num = len(self.list_of_image)
        return self.list_of_image

    def read_info(self):
        '''
        read the sensor record of the car, including steering angle and braking
        :return: steering_list
        '''
        df = pd.read_csv(self.path + 'log.txt')
        df.columns = ['image_name', 'steering_angle', 'brake', 'brake_user', 'brake_computer']
        self.steering_dict = dict(zip(df['image_name'], df['steering_angle']))
        return self.steering_dict

    def shuffle_data(self):
        '''
        shuffle the training set for feed in
        :return: self.shuffled_list
        '''
        self.shuffled_list = self.list_of_image
        random.shuffle(self.shuffled_list)
        return self.shuffled_list

    def resize_image(self, img, size=(80, 160)):
        '''
        resize every single frame from the video to a smaller size
        :return: resized_image
        '''
        return scipy.misc.imresize(img, size)

    def processing(self):
        '''
        the whole process of data processing
        :return: None
        '''
        pass


def main(*args, **kwargs):
    a = Data()
    # print(a.path)
    # print(a.read_datalist())
    # print(a.read_info())
    # print(a.shuffle_data())
    angles = np.array(list(a.steering_dict.values()))
    print(sum(abs(angles)>30)/len(angles))
    plt.show()


if __name__ == '__main__':
    main()