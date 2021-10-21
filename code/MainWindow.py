# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!


import os , sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from SecondWindow import Ui_SecondWindow
from matplotlib.figure import Figure
import cv2
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from data_processing import Data
from run_model import RunModel


class Ui_MainWindow(QWidget, RunModel):
    def __init__(self):
        super().__init__()
        self.window_2 = QtWidgets.QMainWindow()
        self.ui_2 = Ui_SecondWindow()
        self.ui_2.setupUi(self.window_2)
        self.ui_2.go_to_main.clicked.connect(self.show_main)
        self.cwd = os.getcwd()
        self.img = []
        self.dir_choose = []
        self.sess = []
        self.x = []
        self.y = []
        self.keep_prob = []
        self.log = {}  # keys: name of images, value: ground truth of steering angle
        self.read_log()
        self.build_sess()

        # self.parent = parent
        # fig = Figure(figsize=(5, 4), dpi=100)
        # FigureCanvas.__init__(self, fig)
        # # self.setParent(parent)
        #
        # FigureCanvas.setSizePolicy(self,
        #                             QtWidgets.QSizePolicy.Expanding,
        #                             QtWidgets.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

    def show_main(self):
        MainWindow.show()
        tf.reset_default_graph()
        self.build_sess()
        self.window_2.hide()

    def show_window_2(self):
        MainWindow.hide()
        tf.reset_default_graph()
        self.window_2.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(610, 320, 112, 43))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 10, 320, 160))
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(420, 40, 480, 240))
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(570, 300, 160, 160))
        # self.label2.setGeometry(QtCore.QRect(0, 0, 160, 160))

        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label2.setFont(font)
        self.label2.setObjectName("label2")
        self.label3.setFont(font)
        self.label3.setObjectName("label3")
        # self.go_to_window_4 = QtWidgets.QPushButton(self.centralwidget)
        # self.go_to_window_4.setGeometry(QtCore.QRect(690, 140, 92, 23))
        # self.go_to_window_4.setObjectName("go_to_window_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(670, 300, 100, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.Video)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(670, 200, 100, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.setDataset)
        self.go_to_window_2 = QtWidgets.QPushButton(self.centralwidget)
        self.go_to_window_2.setGeometry(QtCore.QRect(670, 400, 100, 50))
        self.go_to_window_2.setObjectName("go_to_window_2")
        self.go_to_window_2.clicked.connect(self.show_window_2)

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 100, 320, 160))
        self.widget.setObjectName("widget")

        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(200, 100, 161, 141))
        self.widget_2.setObjectName("widget_2")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                    "p, li { white-space: pre-wrap; }\n"
                                                    "</style></head><body style=\" font-family:\'Consolas\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
                                                    "<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Real-time Scenarios</p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", " Import Data"))
        self.pushButton.setText(_translate("MainWindow", "Display"))
        self.go_to_window_2.setText(_translate("MainWindow", "Accuracy"))

    # def train(self):   //write training model here

    def read_log(self):
        data = Data()
        self.log = data.steering_dict
        return self.log

    def setDataset(self):
        try:
            self.img = []
            self.dir_choose = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹")  # 起始路径
            # print(self.dir_choose)
            for filename in os.listdir(self.dir_choose):  # listdir的参数是文件夹的路径
                if filename[-1] != 't':
                    self.img.append(filename)
                # print(filename)
            return self.img
        except:  # in case the window is closed and no directory is selected
            pass

    def draw_green(self, img, angle):
        shape = img.shape
        center = (np.floor(shape[0]/2), np.floor(shape[1]/2))
        angle = angle/180*np.pi
        if np.abs(np.sin(angle)) <= 0.5**0.5 and np.cos(angle) > 0:  # up 90 degree
            for i in range(int(np.floor(shape[0]/2))):
                for j in range(shape[1]):
                    if np.abs(j-center[1]+1 - np.tan(angle)*(i-center[0]+1)) < 1:
                        img[i, j, :] = np.zeros([1, 1, 3], dtype=np.uint8)
                        img[i, j, 1] = 255
        elif np.abs(np.sin(angle)) <= 0.5**0.5 and np.cos(angle) < 0:  # down 90 degree
            for i in range(int(np.floor(shape[0]/2)), shape[0]):
                for j in range(shape[1]):
                    if np.abs(j-center[1]+1 - np.tan(angle)*(i-center[0]+1)) < 1:
                        img[i, j, :] = np.zeros([1, 1, 3], dtype=np.uint8)
                        img[i, j, 1] = 255
        elif np.abs(np.cos(angle)) <= 0.5**0.5 and np.sin(angle) > 0:  # left 90 degree
            for j in range(int(np.floor(shape[1]/2))):
                for i in range(shape[0]):
                    if np.abs((j-center[1]+1)/np.tan(angle) - (i-center[0]+1)) < 1:
                        img[i, j, :] = np.zeros([1, 1, 3], dtype=np.uint8)
                        img[i, j, 1] = 255
        else:  # np.abs(np.cos(angle)) < 0.5**0.5 and np.sin(angle) < 0:  # right 90 degree
            for j in range(int(np.floor(shape[1]/2)), shape[1]):
                for i in range(shape[0]):
                    if np.abs((j-center[1]+1)/np.tan(angle) - (i-center[0]+1)) < 1:
                        img[i, j, :] = np.zeros([1, 1, 3], dtype=np.uint8)
                        img[i, j, 1] = 255
        return img

    # def plot_error(self, predict, target):
    #     self.ax = self.figure.add_subplot(111)
    #     # print(dir(self.ax))
    #     self.ax.clear()
    #     self.ax.plot(predict, 'r-')
    #     self.ax.plot(target, 'g-')
    #     # self.ax.hist(data, bins=list(range(0, 31)))
    #     self.ax.set_title('real-time error plot')
    #     self.draw()

    def Video(self):
        label2 = self.label2
        label3 = self.label3


        all_predict = []
        all_target = []

        # print(self.img)

        for i, img in enumerate(self.img):
            # img = cv2.imread('dataset/test_videos/0/%i.png' % i)
            # img = cv2.imread('dataset/test_videos/0/' + )
            # print(img)
            # rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bgrImage = cv2.imread('dataset/test_videos/' + self.dir_choose[78:] + '/' + img)
            rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
            # print(bgrImage)

            convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                             QtGui.QImage.Format_RGB888)
            convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
            pixmap = QPixmap(convertToQtFormat)
            resizeImage = pixmap.scaled(480, 240, QtCore.Qt.KeepAspectRatio)
            QApplication.processEvents()
            label2.setPixmap(resizeImage)
            label2.move(160, 130)
            # self.show()

            predict_angle = -self.predict_one_image(bgrImage)
            all_predict.append(predict_angle)
            # print(predict_angle)  # angle = [[value]]

            steer_wheel = Image.open('steer_beitai3.jpg')
            steer = np.array(steer_wheel.rotate(predict_angle), dtype=np.uint8)
            # steer = steer0[:, 39:-39, :]
            # print(steer.shape)
            steer[steer == 0] = 255

            target_angle = -self.log[img]/10
            steer = self.draw_green(steer, target_angle)
            all_target.append(target_angle)

            # steer = cv2.imread('steer.png')
            # print(np.max(steer))
            # cv2.imshow('steer', steer)
            steerImage = steer
            # print(steerImage.data, steerImage.shape[1], steerImage.shape[0])
            convertToQtFormat_steer = QtGui.QImage(steerImage.data, steerImage.shape[1], steerImage.shape[0],
                                             QtGui.QImage.Format_RGB888)
            convertToQtFormat_steer = QtGui.QPixmap.fromImage(convertToQtFormat_steer)
            pixmap_steer = QPixmap(convertToQtFormat_steer)
            resizeImage_steer = pixmap_steer.scaled(160, 160, QtCore.Qt.KeepAspectRatio)
            # resizeImage_steer = pixmap_steer
            QApplication.processEvents()
            label3.setPixmap(resizeImage_steer)
            label3.move(310, 390)
            # label2.move(160, 160)

            time.sleep(0.01)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
