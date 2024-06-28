import cv2
import numpy as np

class BluePointDetector:
    def __init__(self):
        # Creating mask for blue color
        self.low_blue = np.array([100, 150, 0])
        self.high_blue = np.array([140, 255, 255])

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Creating mask with different color ranges
        mask = cv2.inRange(hsv_img, self.low_blue, self.high_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (-100, -100, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if w * h > 300:
                box = (x, y, x + w, y + h)
                break
        return box
