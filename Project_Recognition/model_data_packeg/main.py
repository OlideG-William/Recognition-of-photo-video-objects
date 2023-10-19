import os
import cv2
import sys

from tkinter.messagebox import *
from tkinter import *
from tkinter import filedialog, messagebox
from Detector import Detector

from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import *


filenameGlobImage = ""
filenameGlobVideo = ""

def Videomain():

    videoPath = filenameGlobVideo

    configPath = os.path.join("model_data_packeg", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data_packeg", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data_packeg", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()



def ImageFuncRecognition():
  image = cv2.imread(filenameGlobImage)
  
  h = image.shape[0]
  w = image.shape[1]

# Path to the weights and model files
  weights = "model_data_packeg/frozen_inference_graph.pb"
  model = "model_data_packeg/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# Load the MobileNet SSD model trained on the COCO dataset
  net = cv2.dnn.readNetFromTensorflow(weights, model)
  class_names = []
  with open("model_data_packeg/coco.names", "r") as f:
     class_names = f.read().strip().split("\n")
    # Create a blob from the image
     blob = cv2.dnn.blobFromImage(image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    # Pass the blog through our network and get predictions on the way out
     net.setInput(blob)
     output = net.forward() # form: (1, 1, 100, 7)
     # Cycle by the number of detected objects
     for detection in output[0, 0, :, :]: # output [0, 0, :, :] looks like: (100, 7)
    # Confidence of the model regarding the detected object
        probability = detection[2]
     # If the reliability of the model is below 30%,
     # We do nothing (continue loop)
        if probability <= 0.4:
            continue
    # Perform elementwise multiplication to obtain
    # Coordinates (x, y) of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
    # Draw the bounding box of the object 
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)
    # Extract the id of the detected object to get its name
        class_id = int(detection[1])
    # Draw the name of the predicted object along with the probability
        label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
        cv2.putText(image, label, (box[0], box[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  cv2.imshow('ImageDetectObject', image)
  cv2.waitKey()




# Open a file selection dialog
def open_file_Photo():
    global filenameGlobImage
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File", 
                                          filetypes=(("JPEG files", "*.jpg",), ("PNG files", "*.png"),
                                                     ("BMP files", "*.bmp"),("all files", "*.*")))
      # (title="Error", message ="Error: file not selected")
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
       filenameGlobImage = filename
       ImageFuncRecognition()
    else: 
       messagebox.showerror(title="Error", message ="Error: Photo file not selected")
       messagebox.showinfo(title = "Info", message="Please select file")
     



    # To open a video file and record it, change the global file name to GlobVide
def open_file_Video():
   global filenameGlobVideo
   filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                         filetypes=(("Video files", "*.mp4 *.avi *.mkv"), ("all files", "*.*")))
  
   if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mkv") or filename.endswith("."):
      filenameGlobVideo = filename
      Videomain()
   else:
       messagebox.showerror(title="Error", message ="Error: Video file not selected")
       messagebox.showinfo(title = "Info", message="Please select file")



# The form of two buttons that open two algorithms for photo/video recognition
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(0, 0, 800, 600)
        self.setWindowTitle('Action Selection Menu')
        self.setWindowIcon(QIcon('model_data_packeg/IconForm/face-scan.png'))
             
        # Event for the background of the form in the form of a photo
        pixmap = QPixmap('model_data_packeg/PhotoBackground/backgrounddetect.jpg')
        pixmap = pixmap.scaled(self.width(), self.height())
        label = QLabel(self)
        label.setPixmap(pixmap)
        label.setGeometry(0, 0, self.width(), self.height())      
        

        # The text of the Inscription on the main form
        self.labl = QLabel('Choose a mode for photo or video processing',self)
        #self.labl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        # Text position Text
        self.setFixedSize(800, 600)
        self.labl.setGeometry(135,50,810,120)
        self.labl.setFont(QtGui.QFont("Comic Sans MS", 15,5))
        self.labl.setStyleSheet("color: yellow; font-weight: bold;")
        self.labl.adjustSize()
        

        #button 1 creation
        btn1 = QPushButton(self)
        btn1.clicked.connect(self.buttonClickedPhoto)
        # btn1.setGeometry(100, 300, 200, 50)
        # Button1's gradient and font color
        btn1.setText('Photo recognition')
        btn1.setFont(QtGui.QFont("Comic Sans MS", 12))
        # Changing the scale of button1 (in length, width)
        btn1.setFixedSize(250, 65)
        gradient = QLinearGradient(0, 0, 177, 48)
        gradient.setColorAt(0, QColor(8, 126, 232))
        gradient.setColorAt(1, QColor(135, 26, 135))
        btn1.setStyleSheet('QPushButton { color: yellow; font-weight: bold; padding: 11px 21px; border-radius: 25px; \
                     background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #087ee8, stop: 1 #871a87); } \
                     QPushButton:hover { background: white; color: #087ee8; border: 2px solid #087ee8; }')
       

        #button 2 creation
        btn2 = QPushButton(self)
        btn2.clicked.connect(self.buttonClickedVideo)
        #btn2.setGeometry(400, 300,200, 50)
        # Button2 gradient and font color
        btn2.setText('Video recognition')
        btn2.setFont(QtGui.QFont("Comic Sans MS", 12))
        # Changing the scale of button2 (in length, width)
        btn2.setFixedSize(250, 65)
        gradient = QLinearGradient(0, 0, 177, 48)
        gradient.setColorAt(0, QColor(8, 126, 232))
        gradient.setColorAt(1, QColor(135, 26, 135))
        btn2.setStyleSheet('QPushButton { color: yellow; font-weight: bold; padding: 11px 21px; border-radius: 25px; \
                     background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #087ee8, stop: 1 #871a87); } \
                     QPushButton:hover { background: white; color: #087ee8; border: 2px solid #087ee8; }')
        

        # Creating a QGridLayout and adding buttons to it
        grid = QGridLayout()
        grid.addWidget(btn1, 0, 0)
        grid.addWidget(btn2, 0, 1)
        self.setLayout(grid)
        # Form call
        self.show()

    def buttonClickedPhoto(self):
        open_file_Photo()

    def buttonClickedVideo(self):
        open_file_Video()
  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

#if __name__ == '__main__':
#    main()
    

    """
        # Event for the background of Forms (in the form of .gif)
        movie = QMovie("model_data_packeg/PhotoBackground/background.gif")
        movie.frameChanged.connect(self.repaint)
        movie.start()
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, 800, 600) # розміри фону
        self.background_label.setMovie(movie)       
        """