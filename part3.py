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

"""/*# Example usage (similar to RedPointDetector)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    bp_detector = BluePointDetector()
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        bluepoint_bbox = bp_detector.detect(frame)
        x, y, x2, y2 = bluepoint_bbox
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if x != -100:
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
"""