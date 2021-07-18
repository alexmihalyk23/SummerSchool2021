import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    bigFrame = cv2.resize(frame, (1400,900))
    ROI = bigFrame[100:400,100:400]
    ROI = cv2.resize(ROI, (0,0), fx=1.5,fy=1.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("frame", frame)
    cv2.imshow("crop", ROI)
    cv2.waitKey(1)
