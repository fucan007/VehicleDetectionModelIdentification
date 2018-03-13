#! /usr/bin/python3
# coding = utf-8

from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog,QDesktopWidget,QHBoxLayout,QVBoxLayout,QInputDialog,QMessageBox
from PyQt5.QtCore import Qt,QThread,qDebug

class PictureWindow(QtWidgets.QWidget):
    def __init__(self):
        super(PictureWindow,self).__init__()
        self.initUi()
        self.center()

    def initUi(self):
        self._diaheight = 400
        self._diawidth = 500
        self.setMinimumHeight(self._diaheight) #设置窗口最小的大小
        self.setMinimumWidth(self._diawidth)
        self.setWindowTitle('Display picture')

        self.InputPictureButton = QtWidgets.QPushButton(self)
        self.InputPictureButton.setObjectName("myButton")
        self.InputPictureButton.setText("选取一张图片")
        self.InputPictureButton.clicked.connect(self.selectPicture)

        self.DisplayPictureButton = QtWidgets.QPushButton(self)
        self.DisplayPictureButton.setObjectName("myButton")
        self.DisplayPictureButton.setText("Display")
        self.DisplayPictureButton.clicked.connect(self.displayPicture)

        self.stopApplicationButton = QtWidgets.QPushButton(self)
        self.stopApplicationButton.setObjectName("myButton")
        self.stopApplicationButton.setText("Exit")
        self.stopApplicationButton.clicked.connect(self.stopApplication)

        self.displayPictureLabel=QtWidgets.QLabel(self)
        self.displayPictureLabel.setAlignment(Qt.AlignCenter)
        #png=QtGui.QPixmap('/home/xiaohui/AI/pyqt_GUI/K-Means_06.png')
        # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
        #self.displayPictureLabel.setPixmap(png)
        #self.displayPictureLabel.move(10,20)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.InputPictureButton)
        self.hbox.addWidget(self.DisplayPictureButton)
        self.hbox.addWidget(self.stopApplicationButton)

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)

        self.vbox.addWidget(self.displayPictureLabel)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)
        qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))


    def selectPicture(self):
        self.pictureName, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          '/home/xiaohui/AI/artificialIntelligenceDocument/MachineLearning_Python/images',
                                                          "Picture Files (*.png)")   #设置文件扩展名过滤,注意
        print (filetype)
        print (self.pictureName)

    def stopApplication(self):
        app.exit(app.exec_())


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def displayPicture(self):

        try:
            png = QtGui.QPixmap(self.pictureName)
        except AttributeError:
            reply = QMessageBox.information(self,
                                    "请先选择一张图片",
                                    "是否现在选择一张图片。",
                                    QMessageBox.Yes | QMessageBox.No)
            #self.displayMessage(reply)
            print (QMessageBox.Yes)
            qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
            if reply == QMessageBox.Yes:
                #qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
                self.selectPicture()
                print ('here!')
        else:
            self.displayPictureLabel.setPixmap(png)

    def displayMessage(self, value):
        '''显示对话框返回值'''
        QMessageBox.information(self, "返回值",   "得到：{}\n\ntype: {}".format(value, type(value)), QMessageBox.Yes | QMessageBox.No)
        #pass

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    myshow=PictureWindow()
    myshow.show()
    sys.exit(app.exec_())
