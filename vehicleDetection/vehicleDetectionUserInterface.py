#! /usr/bin/python3
# coding = utf-8

from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog,QDesktopWidget,QHBoxLayout,QVBoxLayout,QInputDialog,QMessageBox,QTableWidget,QTableWidgetItem
from PyQt5.QtCore import Qt,QThread,qDebug

from location_and_classification_vehicle import location_and_claaification_vehicle

class MyTable(QTableWidget):
    def __init__(self,parent=None):
        super(MyTable, self).__init__(parent)
        self.setWindowTitle("vehicle")
        #self.setWindowIcon(QIcon("male.png"))
        self.resize(500, 200)
        self.setColumnCount(3)
        self.setRowCount(5)
        #设置表格有两行五列。
        self.setColumnWidth(0, 200)
        #self.setColumnWidth(4, 200)
        #self.setRowHeight(0, 100)
        #设置第一行高度为100px，第一列宽度为200px。
        self.table()
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setHorizontalHeaderLabels( ["vehicle_name", "probability", "Position"])
        self.setVerticalHeaderLabels(["第一辆", "第二辆","第三辆","第四辆","第五辆"])


    def table(self):
        self.setItem(0,0,QTableWidgetItem("           你的名字"))
        self.setItem(0,1,QTableWidgetItem("性别"))
        self.setItem(0,2,QTableWidgetItem("出生日期"))
        self.setItem(0,3, QTableWidgetItem("职业"))
        self.setItem(0,4, QTableWidgetItem("收入"))
        #添加表格的文字内容.


class PictureWindow(QtWidgets.QWidget):
    def __init__(self):
        super(PictureWindow,self).__init__()
        self.initUi()
        self.center()
        self.isTraining = False
        self.window2 = MyTable()

    def initUi(self):
        self._diaheight = 800
        self._diawidth = 1000
        self.setMinimumHeight(self._diaheight) #设置窗口最小的大小
        self.setMinimumWidth(self._diawidth)
        self.setWindowTitle('vehicle detection and classification system')

        self.InputPictureButton = QtWidgets.QPushButton(self)
        self.InputPictureButton.setObjectName("myButton")
        self.InputPictureButton.setText("选取一张图片")
        self.InputPictureButton.clicked.connect(self.selectPicture)

        self.TrainPictureButton = QtWidgets.QPushButton(self)
        self.TrainPictureButton.setObjectName("myButton")
        self.TrainPictureButton.setText("Training")
        self.TrainPictureButton.clicked.connect(self.Train)

        self.DisplayPictureButton = QtWidgets.QPushButton(self)
        self.DisplayPictureButton.setObjectName("myButton")
        self.DisplayPictureButton.setText("Detail")
        self.DisplayPictureButton.clicked.connect(self.displayDetailPicture)

        self.DisplayDetailButton = QtWidgets.QPushButton(self)
        self.DisplayDetailButton.setObjectName("myButton")
        self.DisplayDetailButton.setText("display")
        self.DisplayDetailButton.clicked.connect(self.displayPicture)

        self.informationImageButton = QtWidgets.QPushButton(self)
        self.informationImageButton.setObjectName("myButton")
        self.informationImageButton.setText("info")
        self.informationImageButton.clicked.connect(self.informationImage)

        self.stopApplicationButton = QtWidgets.QPushButton(self)
        self.stopApplicationButton.setObjectName("myButton")
        self.stopApplicationButton.setText("Exit")
        self.stopApplicationButton.clicked.connect(self.stopApplication)

        self.displayInputPictureLabel=QtWidgets.QLabel(self)
        self.displayInputPictureLabel.setAlignment(Qt.AlignCenter)

        self.displayOutputPictureLabel = QtWidgets.QLabel(self)
        self.displayOutputPictureLabel.setAlignment(Qt.AlignCenter)
        #png=QtGui.QPixmap('/home/xiaohui/AI/pyqt_GUI/K-Means_06.png')
        # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
        #self.displayInputPictureLabel.setPixmap(png)
        #self.displayInputPictureLabel.move(10,20)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.InputPictureButton)
        self.hbox.addWidget(self.TrainPictureButton)
        self.hbox.addWidget(self.DisplayDetailButton)
        self.hbox.addWidget(self.DisplayPictureButton)
        self.hbox.addWidget(self.informationImageButton)
        self.hbox.addWidget(self.stopApplicationButton)

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)

        self.vbox.addWidget(self.displayInputPictureLabel)
        self.vbox.addWidget(self.displayOutputPictureLabel)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)
        qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))


    def selectPicture(self):
        self.inputPictureName, filetype = QFileDialog.getOpenFileName(self,
                                                              "选取文件",
                                                          '/home/xiaohui/AI/artificialIntelligenceDocument/MachineLearning_Python/images',
                                                          "Picture Files (*.png *.bmp *.jpg *.tif *.GIF *.jpeg )")   #设置文件扩展名过滤,注意
        print (filetype)
        print (self.inputPictureName)

    def stopApplication(self):
        app.exit(app.exec_())


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def displayPicture(self):
        print ('Enter function displayPicture',self.inputPictureName)
        print ('result',self.outputPictureName)
        try:
            inputPixmap = QtGui.QPixmap(self.inputPictureName)
            outputPixmap = QtGui.QPixmap(self.outputPictureName)
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
            print ('here!')
            self.displayInputPictureLabel.setScaledContents(True)
            self.displayInputPictureLabel.setPixmap(inputPixmap)
            self.displayOutputPictureLabel.setScaledContents(True)
            self.displayOutputPictureLabel.setPixmap(outputPixmap)
            print ('here!')

    def displayDetailPicture(self):
        try:
            detailPixmap = QtGui.QPixmap(self.outputDetailPictureName)
            outputPixmap = QtGui.QPixmap(self.outputPictureName)
        except AttributeError:
            reply = QMessageBox.information(self,
                                    "请先选择一张图片",
                                    "是否现在选择一张图片。",
                                    QMessageBox.Yes | QMessageBox.No)
            #self.displayMessage(reply)
            if reply == QMessageBox.Yes:
                #qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
                self.selectPicture()
        else:
            self.displayInputPictureLabel.setScaledContents(True)
            self.displayInputPictureLabel.setPixmap(detailPixmap)


    def informationImage(self):
        try:
            self.inputPictureName
        except AttributeError:
            reply = QMessageBox.information(self,
                                    "NOTIFICATION",
                                    "请先选择一张图片,进行训练,是否现在选择一张图片。",
                                    QMessageBox.Yes | QMessageBox.No)
            #self.displayMessage(reply)
            print (QMessageBox.Yes)
            qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
            if reply == QMessageBox.Yes:
                #qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
                self.selectPicture()
        else:
            if self.isTraining == False:
                QMessageBox.information(self,
                                        "NOTIFICATION",
                                        "请先进行Traing",
                                        QMessageBox.Yes)
            elif self.object_car_num == 0:
                QMessageBox.information(self,
                                        "NOTIFICATION",
                                        "很遗憾！该图片中没有发现任何汽车",
                                        QMessageBox.Yes)
            else:
                print (self.imageFileNameList)
                self.window2.show()

    def displayMessage(self, value):
        '''显示对话框返回值'''
        QMessageBox.information(self, "返回值",   "得到：{}\n\ntype: {}".format(value, type(value)), QMessageBox.Yes | QMessageBox.No)
        #pass

    def Train(self):
        self.isTraining = True
        self.object_car_num,self.outputPictureName,self.outputDetailPictureName,self.imageFileNameList = location_and_claaification_vehicle(self.inputPictureName)
        print ('Traing end!')
        print (self.object_car_num)

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    myshow=PictureWindow()
    myshow.show()
    sys.exit(app.exec_())
