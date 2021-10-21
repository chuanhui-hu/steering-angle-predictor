# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SecondWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QProgressBar, QPushButton
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QBasicTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import run_model
import time
import os

class Ui_SecondWindow(object):
    def setupUi(self, SecondWindow):
        SecondWindow.setObjectName("SecondWindow")
        SecondWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(SecondWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(610, 320, 112, 43))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 10, 201, 61))
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(40, 100, 571, 411))
        # self.widget2 = QtWidgets.QWidget(self.centralwidget)
        # self.widget2.setGeometry(QtCore.QRect(180, 300, 200, 25))
        # self.pbar = QProgressBar(self.widget2)
        # self.pbar.setGeometry(30, 40, 200, 25)
        # self.timer = QBasicTimer()
        # self.step = 0

        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        m = PlotCanvas(self.widget, width=3, height=3)
        m.move(170, 50)

        self.plot_button = QtWidgets.QPushButton(self.centralwidget)
        self.plot_button.setGeometry(QtCore.QRect(600, 200, 120, 50))
        self.plot_button.setObjectName("plot_button")
        self.plot_button.clicked.connect(m.plot)
        self.go_to_main = QtWidgets.QPushButton(self.centralwidget)
        self.go_to_main.setGeometry(QtCore.QRect(600, 350, 120, 50))
        self.go_to_main.setObjectName("go_to_main")


        SecondWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SecondWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")

        SecondWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SecondWindow)
        self.statusbar.setObjectName("statusbar")
        SecondWindow.setStatusBar(self.statusbar)

        self.retranslateUi(SecondWindow)
        QtCore.QMetaObject.connectSlotsByName(SecondWindow)



    def retranslateUi(self, SecondWindow):
        _translate = QtCore.QCoreApplication.translate
        SecondWindow.setWindowTitle(_translate("SecondWindow", "MainWindow"))
        self.label.setText(_translate("SecondWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Accuracy Plot</p></body></html>"))
        self.plot_button.setText(_translate("SecondWindow", "Show Plot"))
        self.go_to_main.setText(_translate("SecondWindow", "Back"))




class PlotCanvas(FigureCanvas, run_model.RunModel):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        run_model.RunModel.__init__(self)
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                    QtWidgets.QSizePolicy.Expanding,
                                    QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot2(self):
        data = [np.random.random() for i in range(25)]
        self.ax = self.figure.add_subplot(111)
        # print(dir(self.ax))
        self.ax.clear()
        self.ax.plot(data, 'r-')
        self.ax.plot([1] * len(data) + data, 'b-')
        # self.ax.hist(data, bins=list(range(0, 31)))
        self.ax.set_title('PyQt Matplotlib Example')
        self.draw()

    def plot(self):
        if not os.path.exists('./loss.txt'):
            print('no loss.txt')
            loss = self.predict_whole()

            np.savetxt('loss.txt', loss)

            self.ax = self.figure.add_subplot(111)
            self.ax.clear()
            self.ax.hist(np.sign(loss) * np.sqrt(np.abs(loss)), bins=np.array(list(range(-50, 51)))/10)
            self.ax.set_title('histogram of errors in degree')
            self.draw()
        else:
            loss = np.loadtxt('loss.txt')

            self.ax = self.figure.add_subplot(111)
            self.ax.clear()
            self.ax.hist(np.sign(loss) * np.sqrt(np.abs(loss)), bins=np.array(list(range(-50, 51)))/10)
            self.ax.set_title('histogram of errors in degree')
            self.draw()





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SecondWindow = QtWidgets.QMainWindow()
    ui = Ui_SecondWindow()
    ui.setupUi(SecondWindow)
    SecondWindow.show()
    sys.exit(app.exec_())

