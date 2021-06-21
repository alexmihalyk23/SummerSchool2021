import pyvirtualcam

import cv2
face_cascade = cv2.CascadeClassifier("haar.xml")
yROI = 80
yMaxROI = 400
xROI = 80
xMaxROI = 560

CenterFace = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    cap = cv2.VideoCapture(0)
    print(f'Using virtual camera: {cam.device}')
    # frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        ret, frame = cap.read()
        ROI = frame.copy()

        CenterROI = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)

        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #
            # if x > 100 and y > 100:
            #     crop_img = frame[y - 100:y + h + 100, x - 100:x + w + 100]
            # else:
            #     crop_img = frame[y:y + h + 100, x + 100:x + w + 100]

            CenterFace = [(x + x + w) // 2, (y + y + h) // 2]
            frame = cv2.circle(frame, CenterFace, 1, (244, 22, 244))
        if not (CenterFace[0] in range(CenterROI[0] - 5, CenterROI[0] + 10) and CenterFace[1] in range(CenterROI[1] - 5,
                                                                                                       CenterROI[
                                                                                                        1] + 10)):
            xROI = xROI + (CenterFace[0] - CenterROI[0])
            yROI = yROI + (CenterFace[1] - CenterROI[1])
            yMaxROI = yMaxROI + (CenterFace[1] - CenterROI[1])
            xMaxROI = xMaxROI + (CenterFace[0] - CenterROI[0])

        if xROI < 0 or yROI < 0 or xMaxROI > 640 or yMaxROI > 480:
            ROI = frame[0:480, 0:640]
        else:
            ROI = frame[yROI:yMaxROI, xROI:xMaxROI]

        cv2.waitKey(0)
        # frame = cv2.resize(frame,(1280,720))
        frame = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        cam.send(cv2.resize(frame,(640,480)))
        cam.sleep_until_next_frame()