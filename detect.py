import torch
import numpy as np
import cv2
from time import time
import os  





class Detector:

    def __init__(self, capture_index, model_name):
        """
        hangi kamerayý kullancaðýmýz, hangi modeli kullanacaðýmýz ekran kartý mý yoksa iþlemci mi kullanacaðýz
        ve bazý deðiþkenlere atama yapýyoruz
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        kameradan görüntü alýyoruz
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Pytorch hub'dan Yolov5 modelini indiriyoruz
        ve bunu modüle geri döndürüyoruz 
        """
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best3.pt', force_reload=True) 

        return model

    def score_frame(self, frame):
        """
        kameradan aldýðý görüntüyü modele sokarak ondan tahmin oraný alýyoruz 
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        classlarýmýzý labela dönüþtürüyoruz.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        aranan objenin hangi konumlar içinde olduðunu buluyoruz.
        """
        dead_counter=0
        live_counter=0
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                

                
                if(labels[i]==1):
                    dead_counter=dead_counter+1
                    # cropped = frame[y1:y2, x1:x2]
                    # gray = cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
                    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,9)
                    # cv2.imshow("winname", th3)
                    # cv2.waitKey(0)
                elif(labels[i]==0):
                    live_counter=live_counter+1
                    # cropped = frame[y1:y2, x1:x2]
                    # gray = cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
                    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,9)
                    # cv2.imshow("winname", th3)
                    # cv2.waitKey(0)
                    
        print("dead: "+str(dead_counter))
        print("live: "+str(live_counter))
        cv2.putText(frame, f'dead:'+str(dead_counter)+' live:'+str(live_counter), (5,frame.shape[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return frame


    def __call__(self):
        
        """
        kameramýzý açarak aranan nesnenin nerede olduðunu hangi nesne olduðunu ve % kaç olasýlýkla onun olduðunu yazýyoruz.
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
      

              
        frame = cv2.imread("2.jpg")
        
            
        frame = cv2.resize(frame, (416,416))
            
        start_time = time()
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)
            
        end_time = time()
            # fps = 1/np.round(end_time - start_time, 2)
            # #print(f"her saniye frame yaz : {fps}")
             
            # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
        cv2.imshow('YOLOv5 Detection', frame)
        cv2.waitKey(0)
 
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break
      
        cap.release()
        cv2.destroyAllWindows()
        
            

        
# yeni bir obje oluþturarak çalýþtýrýyoruz.

detector = Detector(capture_index=0, model_name='best.pt')
detector()