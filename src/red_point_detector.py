import cv2 
import numpy as np 

class RedPointDetector:
    def __init__(self):
        #creating mask for red color
        self.low_red= np.array([160,169,192])
        self.high_red= np.array([179,255,255])

    def detect(self, frame):
        hsv_img= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        #creating mask with diferent color ranges
        mask= cv2.inRange(hsv_img, self.low_red, self.high_red)

        #find countours
        contours , _ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours =  sorted(contours, key=lambda x: cv2.contourArea(x), reverse= True)


        box = (-100,-100,0,0)
        for cnt in contours:
            (x,y,w,h) =cv2.boundingRect(cnt)
            if w*h >300:
                box= (x,y,x+w,y+h)
                break
        return box
    
        
        
