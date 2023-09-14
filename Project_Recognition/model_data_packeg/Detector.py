import cv2
import numpy as np
import time

class Detector:
     def __init__(self, videoPath, configPath, modelPath, classesPath):
         self.VideoPath = videoPath
         self.ConfigPath = configPath
         self.ModelPath = modelPath
         self.ClassesPath = classesPath


         self.net = cv2.dnn_DetectionModel(self.ModelPath, self.ConfigPath)
         self.net.setInputSize(320, 320)
         self.net.setInputScale(1.0/127.5)
         self.net.setInputMean((127.5, 127.5, 127.5))
         self.net.setInputSwapRB(True)

         self.readClasses()

     def readClasses(self):
        with open(self.ClassesPath, 'r') as f:
            self.ClassesList = f.read().splitlines()

            self.ClassesList.insert(0, '__Background__')

            self.colorList = np.random.uniform(low=0,high=255,size=(len(self.ClassesList), 3 ))
           

     def onVideo(self):
         cap = cv2.VideoCapture(self.VideoPath)
         
         if(cap.isOpened() == False):
            print("Error opening file...")
            return
     
         (success, image) = cap.read()
          
         StartTime = 0
                
         while success:
             curentTime = time.time()
             fps = 1/(curentTime - StartTime)
             StartTime = curentTime


             classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

             bboxs = list(bboxs)
             confidences = list(np.array(confidences).reshape(1,-1)[0])
             confidences = list(map(float, confidences))

             bboxIDs = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold=0.8)

             if len(bboxIDs) != 0:
                for i in range(0, len(bboxIDs)):
                    bbox = bboxs[np.squeeze(bboxIDs[i])]
                    classConfidance = confidences[np.squeeze(bboxIDs[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIDs[i])])
                    classLabel = self.ClassesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidance)

                    x,y,w,h = bbox
                    cv2.rectangle(image, (x,y), (x+w, y+h), color= classColor,thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2 )
                    #################
                    lineWidth = min(int(w * 0.3) , int(h * 0.3))
                    
                    cv2.line(image, (x,y), (x+ lineWidth,y), classColor, thickness=5)
                    cv2.line(image, (x,y), (x,y + lineWidth), classColor, thickness=5)

                    cv2.line(image, (x + w,y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x + w,y), (x + w, y + lineWidth), classColor, thickness=5)
                    #########################

                    cv2.line(image, (x,y + h), (x + lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x,y + h), (x , y + h - lineWidth), classColor, thickness=5)

                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x + w,y + h), (x + w , y + h - lineWidth), classColor, thickness=5)

            
             cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2) 
             cv2.namedWindow("DetectorObject", cv2.WINDOW_NORMAL)
             cv2.resizeWindow("DetectorObject", 1100,640)
             cv2.imshow("DetectorObject",image)
           
             key = cv2.waitKey(1) & 0xFF
             if key == ord("q"):
                break
             
             (success, image) = cap.read()
         cv2.destroyAllWindows()

    
             
             