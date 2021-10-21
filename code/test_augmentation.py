from data_processing import Data
import cv2
import numpy as np
import glob
from scipy.misc import imsave


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def add_noise(image):
    image = np.array(image, dtype=np.float32)
    # print(image.shape)
    # print(np.random.random(image.shape).shape)
    image += 20 * np.random.random(image.shape)
    image[image > 255] = 255
    image = np.array(image, dtype=np.uint8)
    return image


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >= 0)] = 1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1]*random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0]*random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def noise():
    train_data = Data()
    dir = './dataset/train/'
    all = glob.glob(dir + '*.png')
    # print(len(all))
    for i, name in enumerate(all):
        # print(name[15:])
        label = train_data.steering_dict[name[16:]]
        # print(label)
        if np.abs(label) > 20:
            img = cv2.imread(name)
            img1 = add_noise(img)
            figure = np.zeros([160, 320, 3])
            for j in range(3):
                figure[:, :, j] = img1[:, :, 2-j]
            imsave(dir + '%i.png' % (60000+i), figure)
            with open(dir + 'log.txt', 'a+') as infile:
                infile.write('%d.png, ' % (60000+i) + '%f, %f, %f, %f\n' % (label, 0, 0, 0))


def brightness():
    train_data = Data()
    dir = './dataset/train/'
    all = glob.glob(dir + '*.png')
    for i, name in enumerate(all):
        # print(name[15:])
        label = train_data.steering_dict[name[16:]]
        if np.abs(label) > 30:
            img = cv2.imread(name)
            img1 = augment_brightness_camera_images(img)
            figure = np.zeros([160, 320, 3])
            for j in range(3):
                figure[:, :, j] = img1[:, :, 2-j]
            imsave(dir + '%i.png' % (120000+i), figure)
            with open(dir + 'log.txt', 'a+') as infile:
                infile.write('%d.png, ' % (120000+i) + '%f, %f, %f, %f\n' % (label, 0, 0, 0))


def main():
    # noise()
    brightness()


if __name__ == '__main__':
    main()


