# import cv2
# import time
# import numpy as np
# import HandTrackingModule as htm
# import math
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#
# ################################
# wCam, hCam = 640, 480
# ################################
#
# cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
# pTime = 0
#
# detector = htm.handDetector(detectionCon=0.7, maxHands=1)
#
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# # volume.GetMute()
# # volume.GetMasterVolumeLevel()
# volRange = volume.GetVolumeRange()
# minVol = volRange[0]
# maxVol = volRange[1]
# vol = 0
# volBar = 400
# volPer = 0
# area = 0
# colorVol = (255, 0, 0)
#
# while True:
#     success, img = cap.read()
#
#     # Find Hand
#     img = detector.findHands(img)
#     lmList, bbox = detector.findPosition(img, draw=True)
#     if len(lmList) != 0:
#         # Filter based on size
#         area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
#         # print(area)
#         if 250 < area < 1000:
#
#             # Find Distance between index and Thumb
#             length, img, lineInfo = detector.findDistance(4, 8, img)
#             # print(length)
#
#             # Convert Volume
#             volBar = np.interp(length, [50, 200], [400, 150])
#             volPer = np.interp(length, [50, 200], [0, 100])
#
#             # Reduce Resolution to make it smoother
#             smoothness = 10
#             volPer = smoothness * round(volPer / smoothness)
#
#             # Check fingers up
#             fingers = detector.fingersUp()
#             # print(fingers)
#
#             # If pinky is down set volume
#             if not fingers[4]:
#                 volume.SetMasterVolumeLevelScalar(volPer / 100, None)
#                 cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
#                 colorVol = (0, 255, 0)
#             else:
#                 colorVol = (255, 0, 0)
#
#     # Drawings
#     cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
#     cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
#     cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
#                 1, (255, 0, 0), 3)
#     cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
#     cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
#                 1, colorVol, 3)
#
#     # Frame rate
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
#                 1, (255, 0, 0), 3)
#
#     cv2.imshow("Img", img)
#     cv2.waitKey(1)


import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 50  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        print("lm",lmList[8])
        x1, y1 = lmList[8]
        x2, y2 = lmList[12]
        print(x1, y1, x2, y2)
     # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
    # # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)
        # cv2.circle(img, (lmList[0]), 15, (0, 0, 222), -1)
        # cv2.circle(img, (lmList[1]), 15, (0, 255, 0), -1)
        # cv2.circle(img, (lmList[2]), 15, (255, 0, 0), -1)
        # cv2.circle(img, (lmList[3]), 15, (255, 0, 255), -1)
        # cv2.circle(img, (lmList[20]), 15, (255, 255, 255), -1)
        # cv2.circle(img, (lmList[16]), 15, (0, 0, 0), -1)
    # # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
    #     # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse

            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. Click mouse if distance short
            if length < 60:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)