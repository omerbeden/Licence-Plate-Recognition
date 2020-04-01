"""
This python file contains main application of licence plate recognition
"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QDesktopWidget
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap,QFont
import sys
import cv2
from  detect_plate import Detection
from  detect_plate_numbers import Detection_Number_Plate
import os
import time


class main_window(QMainWindow):
    def __init__(self):
        super(main_window,self).__init__()
        self.setWindowTitle("Number Plate Reading System")
        self.init_ui()
        self.showMaximized()
                   
    
    def get_images(self):
        """
        This function acquire image in the working directory
        Returns:
            files: a list of image files
        """
        working_dir = os.getcwd()
        self.images_path = os.path.join(working_dir,"images_inference")
        files = []
        for file in os.listdir(self.images_path):
            extension = os.path.splitext(file)[1]
            if ".jpg"  in extension:
                files.append(file)
        return files


    def list_clicked(self):
        """
        This functin show a selected image on the screen
        """
        item = self.list_widget.currentItem()
        self.image_name = item.text()
        self.image_name = self.images_path+"/"+self.image_name
        
        self.label_image = QtWidgets.QLabel(self)
        pixmap_image = QPixmap(self.image_name)
        self.scaled =pixmap_image.scaled(800,700,Qt.KeepAspectRatio)
        self.label_image.setGeometry(0,0,800,700)
        self.label_image.setPixmap(self.scaled) 
        self.label_image.show()
        
       

    def button_clicked(self):
        """
        This function starts the prediction of selected image and calculate elapsed time
        """
        start_time = time.time()
        detect_plate = Detection(self.image_name)
        cropped = detect_plate.run()

        number_plate = Detection_Number_Plate()
        plate , score = number_plate.run(cropped)

        end_time = time.time()
        self.clock_label.setText(str(end_time - start_time))
        self.plaka_label.setText(plate)
        self.score_label.setText(score)

    def init_ui(self):
        """
        This functions has all the widgets about UI
        """
        self.clock_label_vs = QtWidgets.QLabel(self)
        self.clock_label_vs.setText("Time")
        self.clock_label_vs.setFont(QFont('Times',10))
        self.clock_label_vs.setGeometry(900,15,40,40)

        
        self.clock_label = QtWidgets.QLabel(self)
        self.clock_label.setText("-")
        self.clock_label.setGeometry(930,15,40,40)
               
        self.list_widget = QtWidgets.QListWidget(self)    
        self.list_widget.addItems(self.get_images())
        self.list_widget.setGeometry(900,60,100,90)
        self.list_widget.clicked.connect(self.list_clicked)

        self.button = QtWidgets.QPushButton(self)
        self.button.setText("Ok")
        self.button.setGeometry(900,150,50,50)
        self.button.clicked.connect(self.button_clicked)        

        self.plaka_label_vs = QtWidgets.QLabel(self)
        self.plaka_label_vs.setText("Readed Plate")
        self.plaka_label_vs.move(900,300)
        self.plaka_label_vs.setFont(QFont('Times',12))

        self.plaka_label = QtWidgets.QLabel(self)
        self.plaka_label.setText("P")
        self.plaka_label.setFont(QFont('Times',14,QFont.Bold))
        self.plaka_label.move(900,350) 


        self.score_label_vs = QtWidgets.QLabel(self)
        self.score_label_vs.setText("Score")
        self.score_label_vs.setFont(QFont('Times',12,QFont.Bold))
        self.score_label_vs.move(900,400)

        self.score_label = QtWidgets.QLabel(self)
        self.score_label.setText("S")
        self.score_label.setGeometry(900,450,300,50)
         


app = QApplication(sys.argv)
win = main_window()
win.show()
sys.exit(app.exec_())
