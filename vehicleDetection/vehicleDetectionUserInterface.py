#! /usr/bin/python3
# coding = utf-8

from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import QFileDialog,QDesktopWidget,QHBoxLayout,QVBoxLayout,QInputDialog,QMessageBox,QTableWidget,QTableWidgetItem
from PyQt5.QtCore import Qt,QThread,qDebug

import line_profiler    #性能分析工具如果这里报错需要安装sudo pip3 install line_profiler，或者将line_profiler相关的代码删除，删除之后不影响app的使用

from location_and_classification_vehicle import location_and_claaification_vehicle

#创建一个QTableWidget主要用于当检测到汽车，建立一个表格用来存放汽车相关的信息
class MyTable(QTableWidget):
    def __init__(self,parent=None):
        super(MyTable, self).__init__(parent)
        #给表格设置titel
        self.setWindowTitle("vehicle")
        #给表格设置大小
        self.resize(500, 200)
        #设置列数和行数
        self.setColumnCount(5)
        self.setRowCount(5)
        #将第一列和第三列设置为200px
        self.setColumnWidth(0, 200)
        self.setColumnWidth(2, 200)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        #居中显示
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        #给行和列设置名称
        self.setHorizontalHeaderLabels( ["vehicle_name", "probability", "position","image_width","image_height"])
        self.setVerticalHeaderLabels(["First vehicle", "Second vehicles","Third vehicles"," Fourth vehicles "," Fifth vehicles "])


        #将图片相关的信息添加到表哥里
    def updateTableInfo(self,num,imageInfoDictionary,W,H):
        #将之前的表格内容清空
        self.deleteOldItem()
        i = 0
        for vehileInfo in imageInfoDictionary.keys():
            print ('vehileInfo',vehileInfo)
            #调用分割符将名称和概率分开
            vehileName,probability = vehileInfo.split(':')
            Position = imageInfoDictionary[vehileInfo]
            #获取对应的位置信息
            ymin, xmin, ymax, xmax = Position
            xmin = xmin * W
            xmax = xmax * W
            ymin = ymin * H
            ymax = ymax * H
            #设置格式使表格内容更为好看
            disPosition = '(' + str(round(xmin,2)) + ',' + str(round(ymin,2)) + ')' + '  ' + '(' + str(round(xmax,2)) + ',' + str(round(ymax,2)) + ')'
            #将信息填充到表格里
            self.setItem(i, 0, QTableWidgetItem('  ' + vehileName))
            self.setItem(i, 1, QTableWidgetItem(' ' + probability))
            self.setItem(i, 2, QTableWidgetItem(disPosition))
            self.setItem(i, 3, QTableWidgetItem(str(W)))
            self.setItem(i, 4, QTableWidgetItem(str(H)))

            i = i + 1

    def deleteOldItem(self):
        for i in range(5):
            for j in range(5):
                self.setItem(i,j, QTableWidgetItem(None))


class PictureWindow(QtWidgets.QWidget):
    def __init__(self):
        super(PictureWindow,self).__init__()
        #初始化界面
        self.initUi()
        #居中显示
        self.center()
        #设置flag
        self.isTraining = False
        self.window2 = MyTable()

    def initUi(self):
        self._diaheight = 800
        self._diawidth = 1000
        self.setMinimumHeight(self._diaheight) #设置窗口最小的大小
        self.setMinimumWidth(self._diawidth)
        self.setWindowTitle('vehicle detection and classification system')
        #以下都是按钮的初始化，以及和对应的函数进行链接
        self.InputPictureButton = QtWidgets.QPushButton(self)
        self.InputPictureButton.setObjectName("myButton")
        self.InputPictureButton.setText("Selct picture")
        self.InputPictureButton.clicked.connect(self.selectPicture)

        self.TrainPictureButton = QtWidgets.QPushButton(self)
        self.TrainPictureButton.setObjectName("myButton")
        self.TrainPictureButton.setText("Inference")
        self.TrainPictureButton.clicked.connect(self.InferencePicture)

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
        self.informationImageButton.setText("Info")
        self.informationImageButton.clicked.connect(self.informationImage)

        self.stopApplicationButton = QtWidgets.QPushButton(self)
        self.stopApplicationButton.setObjectName("myButton")
        self.stopApplicationButton.setText("Exit")
        self.stopApplicationButton.clicked.connect(self.stopApplication)

        self.displayInputPictureLabel=QtWidgets.QLabel(self)
        self.displayInputPictureLabel.setAlignment(Qt.AlignCenter)

        self.displayOutputPictureLabel = QtWidgets.QLabel(self)
        self.displayOutputPictureLabel.setAlignment(Qt.AlignCenter)

        #对界面进行布局
        self.hbox = QHBoxLayout()#创建水平布局器
        self.hbox.addStretch(1)#增加伸缩量
        self.hbox.addWidget(self.InputPictureButton)#添加按钮
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
        print ('Enter function displayPicture and will display ',self.inputPictureName)
        try:
            # 读取文件转换成pixmap
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
        else:
            self.displayInputPictureLabel.setScaledContents(True)
            self.displayInputPictureLabel.setPixmap(inputPixmap)
            self.displayOutputPictureLabel.setScaledContents(True)
            self.displayOutputPictureLabel.setPixmap(outputPixmap)

    def displayDetailPicture(self):
        try:
            # 读取文件转换成pixmap
            detailPixmap = QtGui.QPixmap(self.outputDetailPictureName)
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
            self.displayInputPictureLabel.setScaledContents(True)#设置图片自适应
            self.displayInputPictureLabel.setPixmap(detailPixmap)#显示图片


    def informationImage(self):
        try:
            self.inputPictureName
        except AttributeError:
            reply = QMessageBox.information(self,
                                    "NOTIFICATION",
                                    "请先选择一张图片,进行训练,是否现在选择一张图片。",
                                    QMessageBox.Yes | QMessageBox.No)
            print (QMessageBox.Yes)
            qDebug('From main thread: %s' % hex(int(QThread.currentThreadId())))
            if reply == QMessageBox.Yes:
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
                print (self.imageInfoDictionary)
                self.window2.updateTableInfo(self.object_car_num,self.imageInfoDictionary,self.im_width, self.im_height)
                self.window2.show()

    def displayMessage(self, value):
        '''显示对话框返回值'''
        QMessageBox.information(self, "返回值",   "得到：{}\n\ntype: {}".format(value, type(value)), QMessageBox.Yes | QMessageBox.No)
        #pass

    def InferencePicture(self):
        self.isTraining = True
        #第二行将图像的路径和名称传入到底层然后取得图像内汽车的数目、将原图处理后的图像的路径和名称等等信息
        prof = line_profiler.LineProfiler(location_and_claaification_vehicle)
        prof.enable()  # 开始性能分析
        self.object_car_num,\
        self.outputPictureName,\
        self.outputDetailPictureName,\
        self.imageInfoDictionary,\
        (self.im_width, self.im_height) = location_and_claaification_vehicle(self.inputPictureName)
        print ('Traing end!')
        prof.disable()  # 停止性能分析
        prof.print_stats(sys.stdout)
if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    myshow=PictureWindow()
    myshow.show()
    sys.exit(app.exec_())
