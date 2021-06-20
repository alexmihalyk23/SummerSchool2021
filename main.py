import pyvirtualcam
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haar.xml")
with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    cap = cv2.VideoCapture(0)
    print(f'Using virtual camera: {cam.device}')
    # frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        ret, frame = cap.read()
        #

        ret, frame = cap.read()

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
        print(faces)
        for x, y, w, h in faces:
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if x > 100 and y > 100:
                frame = frame[y - 100:y + h + 100, x - 100:x + w + 100]
            elif y+h+100 > 480 or x+w+100 >640:
                frame = frame[y:y + h, x:x + w]
            else:
                frame = frame[y:y + h+100, x:x + w+100]

        cv2.waitKey(0)
        # frame = cv2.resize(frame,(1280,720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(cv2.resize(frame,(640,480)))
        cam.sleep_until_next_frame()