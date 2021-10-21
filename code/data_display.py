# Author: Chuanhui
# display the original dataset, by showing the images with corresponding steering angle


import matplotlib.pyplot as plt
import numpy as np
import data_processing
from PIL import Image
import cv2
import time


class DisplayOrigin(data_processing.Data):
    def __init__(self):
        '''
        initialization
        '''
        data_processing.Data.__init__(self, flag='train')

    def display(self):
        '''
        display the dataset
        :return: None
        '''
        images = self.list_of_image
        steer_wheel = Image.open('steering_wheel.jpg')
        for i in range(self.data_num):
            scene = cv2.imread(images[i])
            label = int(self.steering_dict[images[i][16:]])
            # with open('test.txt', 'a+') as infile:
            #     infile.write(images[i] + ', ' + images[i][16:] + '\n')
            # print(images[i])
            # print(images[i][16:])
            out1 = np.array(steer_wheel.rotate(label))
            cv2.imshow('scene', scene)
            cv2.imshow('steer_wheel', out1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)


def main(*args, **kwargs):
    show = DisplayOrigin()
    show.display()


if __name__ == '__main__':
    main()