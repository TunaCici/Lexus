# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'debug.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

import camera
import config

class Ui_ProjectLexusDebugScreen(object):
    def starter(self):
        config.DEBUG_RUNNER = True
        camera.cameraPhotoCapturer(config.PHOTO_NUMBER)
    
    def stopper(self):
        config.DEBUG_RUNNER = False

    def setupUi(self, ProjectLexusDebugScreen):
        ProjectLexusDebugScreen.setObjectName("ProjectLexusDebugScreen")
        ProjectLexusDebugScreen.resize(800, 600)
        ProjectLexusDebugScreen.setMinimumSize(QtCore.QSize(0, 600))
        ProjectLexusDebugScreen.setMaximumSize(QtCore.QSize(800, 16777215))
        self.centralwidget = QtWidgets.QWidget(ProjectLexusDebugScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.baslaButton = QtWidgets.QPushButton(self.centralwidget)
        self.baslaButton.setGeometry(QtCore.QRect(320, 40, 111, 71))
        self.baslaButton.setObjectName("baslaButton")
        self.baslaButton.setStyleSheet("background-color : #ff9f8e")
        self.baslaButton.pressed.connect(self.starter)
        self.durButton = QtWidgets.QPushButton(self.centralwidget)
        self.durButton.setGeometry(QtCore.QRect(320, 120, 111, 71))
        self.durButton.setObjectName("durButton")
        self.durButton.setStyleSheet("background-color : #ff9f8e")
        self.durButton.pressed.connect(self.stopper)
        self.goruntuSecButton = QtWidgets.QPushButton(self.centralwidget)
        self.goruntuSecButton.setGeometry(QtCore.QRect(320, 200, 111, 71))
        self.goruntuSecButton.setObjectName("goruntuSecButton")
        self.goruntuSecButton.setStyleSheet("background-color : #ff9f8e")
        self.modulSituations = QtWidgets.QListWidget(self.centralwidget)
        self.modulSituations.setGeometry(QtCore.QRect(10, 30, 256, 192))
        self.modulSituations.setObjectName("modulSituations")
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.modulSituations.addItem(item)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 94, 16))
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-weight : bold")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 20, 54, 16))
        self.label_2.setObjectName("label_2")
        self.label_2.setStyleSheet("font-weight : bold")
        self.objects = QtWidgets.QListView(self.centralwidget)
        self.objects.setGeometry(QtCore.QRect(440, 370, 341, 192))
        self.objects.setObjectName("objects")
        self.objects.setStyleSheet("font-weight : bold")
        self.logs = QtWidgets.QListWidget(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(20, 370, 361, 192))
        self.logs.setObjectName("logs")
        self.logs.setStyleSheet("font-weight : bold")
        self.aiScreen = QtWidgets.QLabel(self.centralwidget)
        self.aiScreen.setGeometry(QtCore.QRect(510, 40, 261, 211))
        self.aiScreen.setObjectName("aiScreen")
        self.aiScreen.setStyleSheet("font-weight : bold")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(500, 20, 141, 16))
        self.label_4.setObjectName("label_4")
        self.label_4.setStyleSheet("font-weight : bold")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 350, 47, 13))
        self.label_5.setObjectName("label_5")
        self.label_5.setStyleSheet("font-weight : bold")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(440, 350, 131, 16))
        self.label_6.setObjectName("label_6")
        self.label_6.setStyleSheet("font-weight : bold")
        ProjectLexusDebugScreen.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ProjectLexusDebugScreen)
        self.statusbar.setObjectName("statusbar")
        ProjectLexusDebugScreen.setStatusBar(self.statusbar)

        self.retranslateUi(ProjectLexusDebugScreen)
        QtCore.QMetaObject.connectSlotsByName(ProjectLexusDebugScreen)

    def retranslateUi(self, ProjectLexusDebugScreen):
        _translate = QtCore.QCoreApplication.translate
        ProjectLexusDebugScreen.setWindowTitle(_translate("ProjectLexusDebugScreen", "Project Lexus Debug Screen"))
        self.baslaButton.setText(_translate("ProjectLexusDebugScreen", "BASLA"))
        self.durButton.setText(_translate("ProjectLexusDebugScreen", "DUR"))
        self.goruntuSecButton.setText(_translate("ProjectLexusDebugScreen", "GORUNTU SEC"))
        __sortingEnabled = self.modulSituations.isSortingEnabled()
        self.modulSituations.setSortingEnabled(False)
        item = self.modulSituations.item(0)
        item.setText(_translate("ProjectLexusDebugScreen", "KAMERA :  "))
        item = self.modulSituations.item(1)
        item.setText(_translate("ProjectLexusDebugScreen", "TITRESIM : "))
        item = self.modulSituations.item(2)
        item.setText(_translate("ProjectLexusDebugScreen", "YAKINLIK :"))
        item = self.modulSituations.item(3)
        item.setText(_translate("ProjectLexusDebugScreen", "SES : "))
        item = self.modulSituations.item(4)
        item.setText(_translate("ProjectLexusDebugScreen", "KONROLCU : "))
        item = self.modulSituations.item(5)
        item.setText(_translate("ProjectLexusDebugScreen", "YAPAY ZEKA : "))
        self.modulSituations.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("ProjectLexusDebugScreen", "MODUL DURUMU"))
        self.label_2.setText(_translate("ProjectLexusDebugScreen", "ISLEMLER"))
        self.aiScreen.setText(_translate("ProjectLexusDebugScreen", "GORUNTU"))
        self.label_4.setText(_translate("ProjectLexusDebugScreen", "YAPAY ZEKA GORUNTUSU"))
        self.label_5.setText(_translate("ProjectLexusDebugScreen", "LOG"))
        self.label_6.setText(_translate("ProjectLexusDebugScreen", "TESPIT EDILEN OBJELER"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_ProjectLexusDebugScreen()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())