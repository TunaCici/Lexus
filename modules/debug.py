# ----------------------------------------------------------------------------------

# Form implementation generated from reading ui file 'debug.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# ----------------------------------------------------------------------------------

from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from PIL import Image
from PIL.ImageQt import ImageQt
import os
import glob

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import camera
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import camera
    import logger

class DebugScreen(object):
    def logger_start(self):
        if config.IS_LOGGER_RUNNING == True:
            logger.LexusLogger()

        self.file_log = open(config.PROJECT_DIR + "/logs/lexuslogfile.txt")
    
    def start(self):
        config.CAMERA_RUNNING = True
        config.IS_LOGGER_RUNNING = True

        self.item.setText(self._translate("ProjectLexusDebugScreen", "KAMERA :  ACIK"))

        self.obje = camera.Camera()
        self.update()
    
    def close(self):
        config.CAMERA_RUNNING = False
        config.IS_LOGGER_RUNNING = False

        self.item.setText(self._translate("ProjectLexusDebugScreen", "KAMERA :  DEVRE DISI"))

        self.obje.videoCaptureObject.release()
        
        self.obje.photo_no = 0

        self.i = 0
        config.LINE_NUMBER = 0

        self.logs.clear()

        self.file_log.close()
        

    def update(self):
        while(config.CAMERA_RUNNING == True and config.IS_LOGGER_RUNNING == True and self.obje.photo_no != config.PHOTO_NUMBER + 1):
            self.obje.update()
            QtTest.QTest.qWait(100)
            self.files = glob.glob(config.PROJECT_DIR + "/photos/")
            if len(self.files) >= 0:
                self.file = Image.open(config.PROJECT_DIR + "/photos/" + str(self.obje.photo_no) + ".png")
                self.photo = ImageQt(self.file)
                self.aiScreen.resize(config.RESIZE_X,config.RESIZE_Y)
                self.pixmap = QtGui.QPixmap.fromImage(self.photo)
                self.aiScreen.setPixmap(self.pixmap)
                QtTest.QTest.qWait(100)
            self.obje.photo_no = self.obje.photo_no + 1

            if self.obje.photo_no == config.PHOTO_NUMBER:
                self.obje.photo_no = 0

            self.logger_start()

            self.list_Lines = self.file_log.readlines()
            config.LINE_NUMBER = len(self.list_Lines)

            self.i = 0
            self.logs.addItem(self.list_Lines[self.i])
            self.i = self.i + 1

    def goruntu_sec(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName()
        self.path = self.filename[0]
        self.image = Image.open(self.path)
        self.image.show()

    def setupUi(self, ProjectLexusDebugScreen):
        ProjectLexusDebugScreen.setObjectName("ProjectLexusDebugScreen")
        ProjectLexusDebugScreen.resize(1080, 720)
        ProjectLexusDebugScreen.setMinimumSize(QtCore.QSize(0, 600))
        ProjectLexusDebugScreen.setMaximumSize(QtCore.QSize(800, 16777215))
        ProjectLexusDebugScreen.setFixedWidth(900)
        ProjectLexusDebugScreen.setFixedHeight(720)
        ProjectLexusDebugScreen.setStyleSheet("background-color: #ab9191;")
        self.centralwidget = QtWidgets.QWidget(ProjectLexusDebugScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.baslaButton = QtWidgets.QPushButton(self.centralwidget)
        self.baslaButton.setGeometry(QtCore.QRect(320, 40, 111, 71))
        self.baslaButton.setObjectName("baslaButton")
        self.baslaButton.setStyleSheet("background-color : #ff9f8e")
        self.baslaButton.pressed.connect(self.start)
        self.durButton = QtWidgets.QPushButton(self.centralwidget)
        self.durButton.setGeometry(QtCore.QRect(320, 120, 111, 71))
        self.durButton.setObjectName("durButton")
        self.durButton.setStyleSheet("background-color : #ff9f8e")
        self.durButton.pressed.connect(self.close)
        self.goruntuSecButton = QtWidgets.QPushButton(self.centralwidget)
        self.goruntuSecButton.setGeometry(QtCore.QRect(320, 200, 111, 71))
        self.goruntuSecButton.setObjectName("goruntuSecButton")
        self.goruntuSecButton.setStyleSheet("background-color : #ff9f8e")
        self.goruntuSecButton.pressed.connect(self.goruntu_sec)
        self.modulSituations = QtWidgets.QListWidget(self.centralwidget)
        self.modulSituations.setGeometry(QtCore.QRect(10, 30, 200, 150))
        self.modulSituations.setObjectName("modulSituations")
        self.modulSituations.setStyleSheet("background-color : #d2c8c8")
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
        self.objects = QtWidgets.QListWidget(self.centralwidget)
        self.objects.setGeometry(QtCore.QRect(440, 370, 341, 192))
        self.objects.setObjectName("objects")
        self.objects.setStyleSheet("font-weight : bold")
        self.objects.setStyleSheet("background-color : #d2c8c8")
        self.logs = QtWidgets.QListWidget(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(20, 370, 361, 192))
        self.logs.setObjectName("logs")
        self.logs.setStyleSheet("font-weight : bold")
        self.logs.setStyleSheet("background-color : #d2c8c8")
        self.aiScreen = QtWidgets.QLabel(self.centralwidget)
        self.aiScreen.setGeometry(QtCore.QRect(510, 40, 261, 211))
        self.aiScreen.setObjectName("aiScreen")
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
        self._translate = QtCore.QCoreApplication.translate
        ProjectLexusDebugScreen.setWindowTitle(self._translate("ProjectLexusDebugScreen", "Project Lexus Debug Screen"))
        self.baslaButton.setText(self._translate("ProjectLexusDebugScreen", "BASLA"))
        self.durButton.setText(self._translate("ProjectLexusDebugScreen", "DUR"))
        self.goruntuSecButton.setText(self._translate("ProjectLexusDebugScreen", "GORUNTU SEC"))
        __sortingEnabled = self.modulSituations.isSortingEnabled()
        self.modulSituations.setSortingEnabled(False)
        self.item = self.modulSituations.item(0)
        self.item.setText(self._translate("ProjectLexusDebugScreen", "KAMERA :  DEVRE DISI"))
        self.item2 = self.modulSituations.item(1)
        self.item2.setText(self._translate("ProjectLexusDebugScreen", "TITRESIM : DEVRE DISI"))
        self.item3 = self.modulSituations.item(2)
        self.item3.setText(self._translate("ProjectLexusDebugScreen", "YAKINLIK : DEVRE DISI"))
        self.item4 = self.modulSituations.item(3)
        self.item4.setText(self._translate("ProjectLexusDebugScreen", "SES : DEVRE DISI"))
        self.item5 = self.modulSituations.item(4)
        self.item5.setText(self._translate("ProjectLexusDebugScreen", "KONROLCU : DEVRE DISI"))
        self.item6 = self.modulSituations.item(5)
        self.item6.setText(self._translate("ProjectLexusDebugScreen", "YAPAY ZEKA : DEVRE DISI"))
        self.modulSituations.setSortingEnabled(__sortingEnabled)
        self.label.setText(self._translate("ProjectLexusDebugScreen", "MODUL DURUMU"))
        self.label_2.setText(self._translate("ProjectLexusDebugScreen", "ISLEMLER"))
        self.label_4.setText(self._translate("ProjectLexusDebugScreen", "YAPAY ZEKA GORUNTUSU"))
        self.label_5.setText(self._translate("ProjectLexusDebugScreen", "LOG"))
        self.label_6.setText(self._translate("ProjectLexusDebugScreen", "TESPIT EDILEN OBJELER"))