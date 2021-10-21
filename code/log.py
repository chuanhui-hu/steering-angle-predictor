# Author: Chuanhui
# The log.py records the process of training and show the training process


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Log(object):
    def __init__(self, directory, filename):
        '''
        initialization
        :param directory: the directory of the log file
        :param filename: the name of the log file
        '''
        self.path = directory  # the directory of the log file, dtype = str
        self.filename = filename  # the name of the log file, dtype = str
        self.df = []  # the original data
        self.loss = []  # the processed data, shown as the average loss of each epoch
        self.steps_in_epoch = 0
        self.read_file()  # read the file

    def read_file(self):
        '''
        read the file and convert the data into a list
        :return: self.df
        '''
        self.df = pd.read_csv(self.path + self.filename, header=None)
        self.loss = np.zeros([int(self.df[0][len(self.df[0])-1][6:])+1, 1], dtype=np.float32)
        return self.df

    def process_data(self):
        '''
        process the original data and save as the average loss of each epoch
        :return: self.loss
        '''
        loss_index = 2
        if self.filename == 'validation_log.txt':
            loss_index = 1
        for i in range(len(self.df[0])):
            if int(self.df[0][i][6:]) == 0:
                self.steps_in_epoch += 1
            self.loss[int(self.df[0][i][6:])] += float(self.df[loss_index][i][7:])
            # print(int(self.df[0][i][6:]), float(self.df[loss_index][i][7:]))
        self.loss = self.loss/self.steps_in_epoch
        return self.loss

    def draw_plot(self):
        '''
        draw the plot of the training process
        :return: None
        '''
        plt.plot(self.loss)
        plt.xlabel('epoch')
        plt.ylabel(self.filename[:-8] + ' loss')
        plt.show()


def main(*args, **kwargs):
    train_plot = Log('D:\\academic\\Engineering Information Modeling\\project\\code\\model_2019-04-16\\',
                     'validation_log.txt')
    train_plot.process_data()
    train_plot.draw_plot()
    # print(train_plot.loss)


if __name__ == '__main__':
    main()
