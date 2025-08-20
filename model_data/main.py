import os
import cv2
from ultralytics import YOLO

video_path = os.path.join(".", "vid.mp4")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model = YOLO("yolov8n.pt")

while ret:
    cv2.imshow("frame", frame)
    cv2.waitKey(25)
    
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()