# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'debug.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.



from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from PIL import Image
from PIL.ImageQt import ImageQt
import os
import glob
import stat
import sys
import cv2

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import camera
    from modules import logger
    from modules import voice_command
    from modules import ai
else:
    # importing from main and inside the package
    import config
    import camera
    import logger
    import voice_command
    import ai


class DebugScreen(object):
    is_ai_open = False
    is_voice_open = False
    is_camera_open = False
    is_vibration_open = False
    is_ultrasonic_open = False
    is_controller_open = False
    is_logger_open = False

    def ai_status(self,status):
        if status == 'Active':
            self.is_ai_open = True

        elif status == 'Deactive':
            self.is_ai_open = False
    
    def ai_start(self):
        if self.is_ai_open == True:
            self.ai_obj = ai.Lexus_AI()
            self.item4 = self.Sensors.item(3)
            self.item4.setText(self._translate("ProjectLexusDebugScreen", "\t  YAPAY ZEKA : \n\t  ACIK\n"))
            self.item4.setFont(QtGui.QFont("Lucida Console", 14))

    def ai_close(self):
        if self.is_ai_open == False:
            del self.ai_obj
            self.item4 = self.Sensors.item(3)
            self.item4.setText(self._translate("ProjectLexusDebugScreen", "\t  YAPAY ZEKA : \n\t  DEVRE DISI\n"))
            self.item4.setFont(QtGui.QFont("Lucida Console", 14))

    def voice_status(self,status):
        if status == 'Active':
            self.is_voice_open = True

        elif status == 'Deactive':
            self.is_voice_open = False

    def voice_start(self):
        if self.is_voice_open == True:
            self.voice_obj = voice_command.VoiceCommander()
            self.item2 = self.Sensors.item(1)
            self.item2.setText(self._translate("ProjectLexusDebugScreen", "\t  SES : \n\t  ACIK\n"))
            self.item2.setFont(QtGui.QFont("Lucida Console", 14))

    def voice_close(self):
        if self.is_voice_open == False:
            del self.voice_obj
            self.item2 = self.Sensors.item(1)
            self.item2.setText(self._translate("ProjectLexusDebugScreen", "\t  SES : \n\t  DEVRE DISI\n"))
            self.item2.setFont(QtGui.QFont("Lucida Console", 14))

    def camera_status(self,status):
        if status == 'Active':
            self.is_camera_open = True

        elif status == 'Deactive':
            self.is_camera_open = False

    def camera_start(self):
        if self.is_camera_open == True:
            self.obje = camera.Camera()
            self.item1 = self.Sensors.item(0)
            self.item1.setText(self._translate("ProjectLexusDebugScreen", "\n\n\n\t  KAMERA : \n\t  ACIK\n"))
            self.item1.setFont(QtGui.QFont("Lucida Console", 14))

    def camera_close(self):
        if self.is_camera_open == False:
            self.obje.photo_no = 0
            del self.obje.videoCaptureObject
            del self.obje
            self.item1 = self.Sensors.item(0)
            self.item1.setText(self._translate("ProjectLexusDebugScreen", "\n\n\n\t  KAMERA : \n\t  DEVRE DISI\n"))
            self.item1.setFont(QtGui.QFont("Lucida Console", 14))

    def logger_status(self,status):
        if status == 'Active':
            self.is_logger_open = True

        elif status == 'Deactive':
            self.is_logger_open = False

    def logger_start(self):
        if self.is_logger_open == True:
            self.log_obj = logger.LexusLogger()

        self.file_log = open(config.PROJECT_DIR + "/logs/lexuslogfile.txt",encoding='utf-8')

    def logger_close(self):
        if self.is_logger_open == True:
            self.logger.clear()
            self.file_log.close()
    
    def vibration_status(self,status):
        if status == 'Active':
            self.is_logger_open = True

        elif status == 'Deactive':
            self.is_vibration_open = False

    def vibration_close(self):
        if self.is_vibration_open == False:
            self.item3 = self.Sensors.item(2)
            self.item3.setText(self._translate("ProjectLexusDebugScreen", "\t  TITRESIM : \n\t  DEVRE DISI\n"))
            self.item3.setFont(QtGui.QFont("Lucida Console", 14))

    def ultrasonic_status(self,status):
        if status == 'Active':
            self.is_ultrasonic_open = True

        elif status == 'Deactive':
            self.is_ultrasonic_open = False
    
    def ultrasonic_close(self):
        if self.is_ultrasonic_open == False:
            self.item5 = self.Sensors.item(4)
            self.item5.setText(self._translate("ProjectLexusDebugScreen", "\t  UZAKLIK SENSORU : \n\t  DEVRE DISI\n"))
            self.item5.setFont(QtGui.QFont("Lucida Console", 14))

    def conroller_status(self,status):
        if status == 'Active':
            self.is_controller_open = True

        elif status == 'Deactive':
            self.is_controller_open = False

    def contoller_close(self):
        if self.is_controller_open == False:
            self.item6 = self.Sensors.item(5)
            self.item6.setText(self._translate("ProjectLexusDebugScreen", "\t  KONTROLCU : \n\t  DEVRE DISI\n"))
            self.item6.setFont(QtGui.QFont("Lucida Console", 14))

    def save_function(self):
        try:
            self.obje.save()

        except:
            print("Camera is not opened.")

    def start(self):
        __sortingEnabled = self.Sensors.isSortingEnabled()
        self.Sensors.setSortingEnabled(False)
        
        self.voice_status('Active')
        self.voice_start()
        self.ai_status('Active')
        self.ai_start()
        self.camera_status('Active')
        self.camera_start()
        self.logger_status('Active')
        
        if self.is_vibration_open == True:
            self.item3 = self.Sensors.item(2)
            self.item3.setText(self._translate("ProjectLexusDebugScreen", "\t  TITRESIM : \n\t  ACIK\n"))
            self.item3.setFont(QtGui.QFont("Lucida Console", 14))

        else:
            self.item3 = self.Sensors.item(2)
            self.item3.setText(self._translate("ProjectLexusDebugScreen", "\t  TITRESIM : \n\t  DEVRE DISI\n"))
            self.item3.setFont(QtGui.QFont("Lucida Console", 14))
        
        if self.is_ultrasonic_open == True:
            self.item5 = self.Sensors.item(4)
            self.item5.setText(self._translate("ProjectLexusDebugScreen", "\t  UZAKLIK SENSORU : \n\t  ACIK\n"))
            self.item5.setFont(QtGui.QFont("Lucida Console", 14))

        else:
            self.item5 = self.Sensors.item(4)
            self.item5.setText(self._translate("ProjectLexusDebugScreen", "\t  UZAKLIK SENSORU : \n\t  DEVRE DISI\n"))
            self.item5.setFont(QtGui.QFont("Lucida Console", 14))
        
        if self.is_controller_open == True:
            self.item6 = self.Sensors.item(5)
            self.item6.setText(self._translate("ProjectLexusDebugScreen", "\t  KONTROLCU : \n\t  ACIK\n"))
            self.item6.setFont(QtGui.QFont("Lucida Console", 14))

        else:
            self.item6 = self.Sensors.item(5)
            self.item6.setText(self._translate("ProjectLexusDebugScreen", "\t  KONTROLCU : \n\t  DEVRE DISI\n"))
            self.item6.setFont(QtGui.QFont("Lucida Console", 14))
        
        self.Sensors.setSortingEnabled(__sortingEnabled)

        self.update()
    
    def close(self):
        __sortingEnabled = self.Sensors.isSortingEnabled()
        self.Sensors.setSortingEnabled(False)
        
        try:
            self.ai_status('Deactive')
            self.ai_close()

        except:
            print("AI Closing Failed")
            
        try:
            self.voice_status('Deactive')
            self.voice_close()

        except:
            print("Voice Closing Failed")

        try:
            self.camera_status('Deactive')
            self.camera_close()

        except:
            print("Camera Closing Failed")

        try:
            self.logger_status('Deactive')
            self.logger_close()

        except:
            print("Log Closing Failed")

        try:
            self.vibration_status('Deactive')
            self.vibration_close()

        except:
            print("Vibration Closing Failed")

        try:
            self.ultrasonic_status('Deactive')
            self.ultrasonic_close()

        except:
            print("Ultrasonic Sensor Closing Failed")

        try:
            self.conroller_status('Deactive')
            self.contoller_close()

        except:
            print("Controller Closing Failed")
            
        self.Sensors.setSortingEnabled(__sortingEnabled)

        self.i = 0
        config.LINE_NUMBER = 0

        try:
            self.files = glob.glob(config.PROJECT_DIR + "/photos/")

            for file in self.files:
                os.chmod(file, mode=stat.S_IWRITE)
                os.remove(file)

        except PermissionError as p:
            print("Permission Error Occured...")

    def get_picture(self):
        dim = (521,461)
        self.picture_list[-1] = cv2.resize(self.picture_list[-1],dim)
        return self.picture_list[-1]

    def update(self):
        while(self.is_camera_open == True and self.obje.photo_no != config.PHOTO_NUMBER + 1):
            self.obje.update()
            QtTest.QTest.qWait(100)

            self.ambulance = 0
            self.bench = 0
            self.bicycle = 0
            self.bus = 0
            self.car = 0
            self.cat = 0
            self.chair = 0
            self.couch = 0
            self.dog = 0
            self.motorcycle = 0
            self.person = 0
            self.stop_sign = 0
            self.taxi = 0
            self.traffic_light = 0
            self.traffic_sign = 0

            self.picture = self.obje.get_frame()

            self.picture,self.detection_list = self.ai_obj.update(self.picture)

            self.picture_list = list()
            self.picture_list.append(self.picture)
            
            height, width, channel = self.get_picture().shape
            bytesPerLine = 3 * width
            self.qImg = QtGui.QImage(self.get_picture().data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

            self.pixmap = QtGui.QPixmap(self.qImg)
            self.camera.setPixmap(self.pixmap)
            self.camera.resize(width,height)

            for i in self.detection_list:
                if i[0] == "ambulance":
                    self.ambulance = self.ambulance + 1

                if i[0] == "bench":
                    self.bench = self.bench + 1

                if i[0] == "bicycle":
                    self.bicycle = self.bicycle + 1

                if i[0] == "bus":
                    self.bus = self.bus + 1

                if i[0] == "car":
                    self.car = self.car + 1

                if i[0] == "cat":
                    self.cat = self.cat + 1

                if i[0] == "chair":
                    self.chair = self.chair + 1

                if i[0] == "couch":
                    self.couch = self.couch + 1

                if i[0] == "dog":
                    self.dog = self.dog + 1

                if i[0] == "motorcycle":
                    self.motorcycle = self.motorcycle + 1

                if i[0] == "person":
                    self.person = self.person + 1

                if i[0] == "stop sign":
                    self.stop_sign = self.stop_sign + 1

                if i[0] == "taxi":
                    self.taxi = self.taxi + 1

                if i[0] == "traffic light":
                    self.traffic_light = self.traffic_light + 1

                if i[0] == "traffic sign":
                    self.traffic_sign = self.traffic_sign + 1

            __sort = self.Objects.isSortingEnabled()
            self.Objects.setSortingEnabled(False)
            self.x1 = self.Objects.item(0)
            self.x1.setText(self._translate("ProjectLexusDebugScreen","Ambulance: " + str(self.ambulance)))
            self.x1.setFont(QtGui.QFont("Arial", 11))
            self.x2 = self.Objects.item(1)
            self.x2.setText(self._translate("ProjectLexusDebugScreen","Bench: " + str(self.bench)))
            self.x2.setFont(QtGui.QFont("Arial", 11))
            self.x3 = self.Objects.item(2)
            self.x3.setText(self._translate("ProjectLexusDebugScreen","Bicycle: " + str(self.bicycle)))
            self.x3.setFont(QtGui.QFont("Arial", 11))
            self.x4 = self.Objects.item(3)
            self.x4.setText(self._translate("ProjectLexusDebugScreen","Bus: " + str(self.bus)))
            self.x4.setFont(QtGui.QFont("Arial", 11))
            self.x5 = self.Objects.item(4)
            self.x5.setText(self._translate("ProjectLexusDebugScreen","Car: " + str(self.car)))
            self.x5.setFont(QtGui.QFont("Arial", 11))
            self.x6 = self.Objects.item(5)
            self.x6.setText(self._translate("ProjectLexusDebugScreen","Cat: "  + str(self.cat)))
            self.x6.setFont(QtGui.QFont("Arial", 11))
            self.x7 = self.Objects.item(6)
            self.x7.setText(self._translate("ProjectLexusDebugScreen","Chair: " + str(self.chair)))
            self.x7.setFont(QtGui.QFont("Arial", 11))
            self.x8 = self.Objects.item(7)
            self.x8.setText(self._translate("ProjectLexusDebugScreen","Couch: " + str(self.couch)))
            self.x8.setFont(QtGui.QFont("Arial", 11))
            self.x9 = self.Objects.item(8)
            self.x9.setText(self._translate("ProjectLexusDebugScreen","Dog: " + str(self.dog)))
            self.x9.setFont(QtGui.QFont("Arial", 11))
            self.x10 = self.Objects.item(9)
            self.x10.setText(self._translate("ProjectLexusDebugScreen","Motorcycle: " + str(self.motorcycle)))
            self.x10.setFont(QtGui.QFont("Arial", 11))
            self.x11 = self.Objects.item(10)
            self.x11.setText(self._translate("ProjectLexusDebugScreen","People: "  + str(self.person)))
            self.x11.setFont(QtGui.QFont("Arial", 11))
            self.x12 = self.Objects.item(11)
            self.x12.setText(self._translate("ProjectLexusDebugScreen","Stop Sign: " + str(self.stop_sign)))
            self.x12.setFont(QtGui.QFont("Arial", 11))
            self.x13 = self.Objects.item(12)
            self.x13.setText(self._translate("ProjectLexusDebugScreen","Taxi: " + str(self.taxi)))
            self.x13.setFont(QtGui.QFont("Arial", 11))
            self.x14 = self.Objects.item(13)
            self.x14.setText(self._translate("ProjectLexusDebugScreen","Traffic Light: " + str(self.traffic_light)))
            self.x14.setFont(QtGui.QFont("Arial", 11))
            self.x15 = self.Objects.item(14)
            self.x15.setText(self._translate("ProjectLexusDebugScreen","Traffic Sign: " + str(self.traffic_sign)))
            self.x15.setFont(QtGui.QFont("Arial", 11))
            self.Objects.setSortingEnabled(__sort)

            QtTest.QTest.qWait(100)
            
            if self.is_camera_open == True:
                self.obje.photo_no = self.obje.photo_no + 1

                if self.obje.photo_no == config.PHOTO_NUMBER:
                    self.obje.photo_no = 0
                    self.obje.frame_list.clear()

                QtTest.QTest.qWait(100)

                self.logger_start()
                self.list_Lines = self.file_log.readlines()
                config.LINE_NUMBER = len(self.list_Lines)

                self.i = 0
                for i in range(config.LINE_NUMBER):
                    self.logger.addItem(self.list_Lines[self.i])
                    self.i = self.i + 1

    def goruntu_sec(self):
        try:
            self.filename = QtWidgets.QFileDialog.getOpenFileName(directory=config.PROJECT_DIR + "/photos/")
            self.path = self.filename[0]
            self.image = Image.open(self.path)
            self.image.show()

        except:
            print("No Picture Selected")

    def setupUi(self, ProjectLeXuSDebugScreen):
        ProjectLeXuSDebugScreen.setObjectName("ProjectLeXuSDebugScreen")
        ProjectLeXuSDebugScreen.resize(1037, 779)
        self.centralwidget = QtWidgets.QWidget(ProjectLeXuSDebugScreen)
        self.centralwidget.setObjectName("centralwidget")
        ProjectLeXuSDebugScreen.setStyleSheet("background-color : #c3c3c3")
        self.Sensors = QtWidgets.QListWidget(self.centralwidget)
        self.Sensors.setGeometry(QtCore.QRect(720, 0, 311, 141))
        self.Sensors.resize(311,500)
        self.Sensors.setObjectName("Sensors")
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.Sensors.addItem(item)
        self.Objects = QtWidgets.QListWidget(self.centralwidget)
        self.Objects.setGeometry(QtCore.QRect(720, 460, 311, 291))
        self.Objects.setObjectName("Objects")
        x1 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x1)
        x2 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x2)
        x3 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x3)
        x4 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x4)
        x5 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x5)
        x6 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x6)
        x7 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x7)
        x8 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x8)
        x9 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x9)
        x10 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x10)
        x11 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x11)
        x12 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x12)
        x13 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x13)
        x14 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x14)
        x15 = QtWidgets.QListWidgetItem()
        self.Objects.addItem(x15)
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(0, 0, 521, 461))
        self.camera.setObjectName("camera")
        self.logger = QtWidgets.QListWidget(self.centralwidget)
        self.logger.setGeometry(QtCore.QRect(0, 460, 721, 291))
        self.logger.setObjectName("logger")
        self.logger.setStyleSheet("font-weight : bold")
        self.baslaButton = QtWidgets.QPushButton(self.centralwidget)
        self.baslaButton.setGeometry(QtCore.QRect(530, 0, 181, 71))
        self.baslaButton.setObjectName("baslaButton")
        self.baslaButton.pressed.connect(self.start)
        self.baslaButton.setStyleSheet("background-color : #8f495f")
        self.durButton = QtWidgets.QPushButton(self.centralwidget)
        self.durButton.setGeometry(QtCore.QRect(530, 90, 181, 71))
        self.durButton.setObjectName("durButton")
        self.durButton.pressed.connect(self.close)
        self.durButton.setStyleSheet("background-color : #8f495f")
        self.goruntuSecButton = QtWidgets.QPushButton(self.centralwidget)
        self.goruntuSecButton.setGeometry(QtCore.QRect(530, 180, 181, 71))
        self.goruntuSecButton.setObjectName("goruntuSecButton")
        self.goruntuSecButton.pressed.connect(self.goruntu_sec)
        self.goruntuSecButton.setStyleSheet("background-color : #8f495f")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(530, 270, 181, 71))
        self.saveButton.setObjectName("baslaButton")
        self.saveButton.pressed.connect(self.save_function)
        self.saveButton.setStyleSheet("background-color : #8f495f")
        self.quitButton = QtWidgets.QPushButton(self.centralwidget)
        self.quitButton.setGeometry(QtCore.QRect(530, 360, 181, 71))
        self.quitButton.setObjectName("quitButton")
        self.quitButton.pressed.connect(sys.exit)
        self.quitButton.setStyleSheet("background-color : #8f495f")
        ProjectLeXuSDebugScreen.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ProjectLeXuSDebugScreen)
        self.statusbar.setObjectName("statusbar")
        ProjectLeXuSDebugScreen.setStatusBar(self.statusbar)

        self.retranslateUi(ProjectLeXuSDebugScreen)
        QtCore.QMetaObject.connectSlotsByName(ProjectLeXuSDebugScreen)

    def retranslateUi(self, ProjectLeXuSDebugScreen):
        self._translate = QtCore.QCoreApplication.translate
        ProjectLeXuSDebugScreen.setWindowTitle(self._translate("ProjectLeXuSDebugScreen", "Project LeXuS Debug Screen"))
        __sortingEnabled = self.Sensors.isSortingEnabled()
        self.Sensors.setSortingEnabled(False)
        self.item1 = self.Sensors.item(0)
        self.item1.setText(self._translate("ProjectLexusDebugScreen", "\n\n\n\t  KAMERA : \n\t  DEVRE DISI\n"))
        self.item1.setFont(QtGui.QFont("Lucida Console", 14))
        self.item2 = self.Sensors.item(1)
        self.item2.setText(self._translate("ProjectLexusDebugScreen", "\t  SES : \n\t  DEVRE DISI\n"))
        self.item2.setFont(QtGui.QFont("Lucida Console", 14))
        self.item3 = self.Sensors.item(2)
        self.item3.setText(self._translate("ProjectLexusDebugScreen", "\t  TITRESIM : \n\t  DEVRE DISI\n"))
        self.item3.setFont(QtGui.QFont("Lucida Console", 14))
        self.item4 = self.Sensors.item(3)
        self.item4.setText(self._translate("ProjectLexusDebugScreen", "\t  YAPAY ZEKA : \n\t  DEVRE DISI\n"))
        self.item4.setFont(QtGui.QFont("Lucida Console", 14))
        self.item5 = self.Sensors.item(4)
        self.item5.setText(self._translate("ProjectLexusDebugScreen", "\t  UZAKLIK SENSORU : \n\t  DEVRE DISI\n"))
        self.item5.setFont(QtGui.QFont("Lucida Console", 14))
        self.item6 = self.Sensors.item(5)
        self.item6.setText(self._translate("ProjectLexusDebugScreen", "\t  KONTROLCU : \n\t  DEVRE DISI\n"))
        self.item6.setFont(QtGui.QFont("Lucida Console", 14))
        self.Sensors.setSortingEnabled(__sortingEnabled)

        __sort = self.Objects.isSortingEnabled()
        self.Objects.setSortingEnabled(False)
        self.x1 = self.Objects.item(0)
        self.x1.setText(self._translate("ProjectLexusDebugScreen","Ambulance: 0"))
        self.x1.setFont(QtGui.QFont("Arial", 11))
        self.x2 = self.Objects.item(1)
        self.x2.setText(self._translate("ProjectLexusDebugScreen","Bench: 0"))
        self.x2.setFont(QtGui.QFont("Arial", 11))
        self.x3 = self.Objects.item(2)
        self.x3.setText(self._translate("ProjectLexusDebugScreen","Bicycle: 0"))
        self.x3.setFont(QtGui.QFont("Arial", 11))
        self.x4 = self.Objects.item(3)
        self.x4.setText(self._translate("ProjectLexusDebugScreen","Bus: 0"))
        self.x4.setFont(QtGui.QFont("Arial", 11))
        self.x5 = self.Objects.item(4)
        self.x5.setText(self._translate("ProjectLexusDebugScreen","Car: 0"))
        self.x5.setFont(QtGui.QFont("Arial", 11))
        self.x6 = self.Objects.item(5)
        self.x6.setText(self._translate("ProjectLexusDebugScreen","Cat: 0"))
        self.x6.setFont(QtGui.QFont("Arial", 11))
        self.x7 = self.Objects.item(6)
        self.x7.setText(self._translate("ProjectLexusDebugScreen","Chair: 0"))
        self.x7.setFont(QtGui.QFont("Arial", 11))
        self.x8 = self.Objects.item(7)
        self.x8.setText(self._translate("ProjectLexusDebugScreen","Couch: 0"))
        self.x8.setFont(QtGui.QFont("Arial", 11))
        self.x9 = self.Objects.item(8)
        self.x9.setText(self._translate("ProjectLexusDebugScreen","Dog: 0"))
        self.x9.setFont(QtGui.QFont("Arial", 11))
        self.x10 = self.Objects.item(9)
        self.x10.setText(self._translate("ProjectLexusDebugScreen","Motorcycle: 0"))
        self.x10.setFont(QtGui.QFont("Arial", 11))
        self.x11 = self.Objects.item(10)
        self.x11.setText(self._translate("ProjectLexusDebugScreen","People: 0"))
        self.x11.setFont(QtGui.QFont("Arial", 11))
        self.x12 = self.Objects.item(11)
        self.x12.setText(self._translate("ProjectLexusDebugScreen","Stop Sign: 0"))
        self.x12.setFont(QtGui.QFont("Arial", 11))
        self.x13 = self.Objects.item(12)
        self.x13.setText(self._translate("ProjectLexusDebugScreen","Taxi: 0"))
        self.x13.setFont(QtGui.QFont("Arial", 11))
        self.x14 = self.Objects.item(13)
        self.x14.setText(self._translate("ProjectLexusDebugScreen","Traffic Light: 0"))
        self.x14.setFont(QtGui.QFont("Arial", 11))
        self.x15 = self.Objects.item(14)
        self.x15.setText(self._translate("ProjectLexusDebugScreen","Traffic Sign: 0"))
        self.x15.setFont(QtGui.QFont("Arial", 11))
        self.Objects.setSortingEnabled(__sort)

        self.baslaButton.setText(self._translate("ProjectLeXuSDebugScreen", "BASLA"))
        self.durButton.setText(self._translate("ProjectLeXuSDebugScreen", "DUR"))
        self.goruntuSecButton.setText(self._translate("ProjectLeXuSDebugScreen", "GORUNTU SEC"))
        self.saveButton.setText(self._translate("ProjectLeXuSDebugScreen", "KAYDET"))
        self.quitButton.setText(self._translate("ProjectLeXuSDebugScreen", "KAPAT"))