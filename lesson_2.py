# # сегодня познакомимся с каскадом хаара
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# face_cascade = cv2.CascadeClassifier("haar.xml")
#
# while True:
#     ret, frame = cap.read()
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
#     print(faces)
#     for x, y, w, h in faces:
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
#
#     cv2.imshow("lesson_2", frame)
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()

#
# import cv2
#
# cap = cv2.VideoCapture(0)
#
#
# face_cascade = cv2.CascadeClassifier("haar.xml")
# yROI = 80
# yMaxROI = 400
# xROI = 80
# xMaxROI = 560
# #вычисляем центр лица, так как изначально мы не знаем координат лица задаим стандартное значение центра
# #нашего bboxа. Центр высчитывается как сумма координат x деленное на 2 и y деленное на  2
# CenterFace = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
# while True:
#     ret, frame = cap.read()
# #    метод copy берет копию кадра
#     ROI = frame.copy()
# #    теперь высчитываем центр нашего прямоугольника "интереса"
#     CenterROI = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
# #    теперь получаем данные об обнаруженном лице
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
# # в нашей переменной теперь хранятся хначения координат лица
#     for x, y, w, h in faces:
# #        высчитываем центр лица
#         CenterFace = [(x + x + w) // 2, (y + y + h) // 2]
#
#         frame = cv2.circle(frame, CenterFace, 1, (244, 22, 244))
# #       Далее мы сделаем проверку если центр лица находится не в диапазоне центра ROI тогда
#     if not (CenterFace[0] in range(CenterROI[0] - 5, CenterROI[0] + 10) and CenterFace[1] in range(CenterROI[1] - 5,
#                                                                                                    CenterROI[
#                                                                                                        1] + 10)):
# #       изменяем кроп кадра, высчитывая разность центров
#         xROI = xROI + (CenterFace[0] - CenterROI[0])
#         yROI = yROI + (CenterFace[1] - CenterROI[1])
#         yMaxROI = yMaxROI + (CenterFace[1] - CenterROI[1])
#         xMaxROI = xMaxROI + (CenterFace[0] - CenterROI[0])
#
#     # if xMaxROI > 640 or yMaxROI > 480:
#     #     ROI = frame[xROI:400, yROI:560] #?
#
#     # print(xROI, yROI, xROIpred, yROIpred)
#     if xROI < 0 or yROI<0 or xMaxROI > 640 or yMaxROI>480:
#         ROI = frame[0:480, 0:640]
#     else:
#         ROI = frame[yROI:yMaxROI, xROI:xMaxROI]
#
#     frame = cv2.rectangle(frame, (xROI, yROI), (xMaxROI, yMaxROI), (0, 0, 255), 3)
#     frame = cv2.circle(frame, CenterROI, 1, (244, 22, 244))
#
#     cv2.imshow("test", cv2.resize(frame, (640, 480)))
#     cv2.imshow("test1", cv2.resize(ROI, (640,480)))
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()


# наконец мы научимся рабоать с нейросетями
# мы воспользуемся библиотекой medipipe от Google
# вы наверняка ее уже встречали если пользовались google meet
# удаление заднего фона происходит благодаря данной библеотеке
# https://mediapipe.dev/
# возьмем пример с официального сайта с примером распознавания лица

# import cv2
# import mediapipe as mp
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# # For webcam input:
# cap = cv2.VideoCapture(0)
## тут в FaceDetection можно передать параметр точноти распознавания лица
# with mp_face_detection.FaceDetection() as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = face_detection.process(image)
#
#     # Draw the face detection annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detections:
#       for detection in results.detections:
#         mp_drawing.draw_detection(image, detection)
#     cv2.imshow('MediaPipe Face Detection', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()







# теперь познакомимся с более удобной библиотекой для работы с mediapipe
#cvzone pip install cvzone
#из библиотеки cvzone импортируем FaceDetector, который мы изучили ранее
# from cvzone.FaceDetectionModule import FaceDetector
# import cv2
#
# cap = cv2.VideoCapture(0)
# #инициализируем FaceDetector, который принимает параметр с какой точностью будет распознаваться лицо
# detector = FaceDetector(0.6)
# while True:
#     ret, frame = cap.read()
#     # теперь вызовим метод findFaces, передадим параметры изображение и зададим отрисовку
#     frame, detectionData = detector.findFaces(frame, draw=False)
#     # нам вернут уже обработанный кадр и данные о лице, где будет наше лицо
#     print(detectionData)
#     if len(detectionData) !=0:
#         frame = cv2.circle(frame, detectionData[0].get("center"), 10,(0,0,255),-1)
#         frame = cv2.rectangle(frame, detectionData[0].get("bbox"), (255,0,0))
#         print(detectionData[0].get("score")[0])
#         print(detectionData[0].get("bbox")[0])
#         cv2.putText(frame, str(f'{int(detectionData[0].get("score")[0] * 100)}%'), (detectionData[0].get("bbox")[0],detectionData[0].get("bbox")[1]),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2)
#
#     cv2.imshow("lesson_2", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break



# теперь давайте сравним точность каскада хаара и нейронной сети для распознаввния лиц
# для этого склдеем два наших кадра
# с нейронной сетью и каскадом хаара
# import cv2
# from cvzone.FaceDetectionModule import FaceDetector
# import cvzone
# cap = cv2.VideoCapture(0)
#
# face_cascade = cv2.CascadeClassifier("haar.xml")
# detector = FaceDetector(0.6)
# # также сделаем счетчик кадров в секунду
# fps = cvzone.FPS()
# while True:
#     ret, frame = cap.read()
#     frame_nn = frame.copy()
#     frame_nn, bbox = detector.findFaces(frame_nn, draw=True)
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
#     print(faces)
#
#     for x, y, w, h in faces:
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
#     # мы воспользуемся методом stackImages который принимает наши изображения количество колонок и размером,
#     # мы возьмем стандартный размер 1 к 1
#     StackedImages = cvzone.stackImages([frame, frame_nn], 2,1)
#     #для подсчета fps и отображения будем использовать метод update
#     fps_1, StackedImages = fps.update(StackedImages, color=(0,255,255))
#     print(fps_1)
#     cv2.imshow("lesson_2", StackedImages)
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()


# Задание сделать отслеживание лица, которое мы сделали с помощью каскада хаара, с помощью нейронной сети

# import cv2
# from cvzone.FaceDetectionModule import FaceDetector
# cap = cv2.VideoCapture(0)
#
#
# detector = FaceDetector(0.6)
# yROI = 80
# yMaxROI = 400
# xROI = 80
# xMaxROI = 560
# #вычисляем центр лица, так как изначально мы не знаем координат лица задаим стандартное значение центра
# #нашего bboxа. Центр высчитывается как сумма координат x деленное на 2 и y деленное на  2
# CenterFace = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
# while True:
#     ret, frame = cap.read()
# #    метод copy берет копию кадра
#     ROI = frame.copy()
# #    теперь высчитываем центр нашего прямоугольника "интереса"
#     CenterROI = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
# #    теперь получаем данные об обнаруженном лице
#     frame, detectionData = detector.findFaces(frame, draw=True)
#     if len(detectionData) != 0:
#         x,y,w,h =  detectionData[0].get("bbox")
#         print(x,y,w,h)
#         CenterFace = [(x + x + w) // 2, (y + y + h) // 2]
#
#
#     #       Далее мы сделаем проверку если центр лица находится не в диапазоне центра ROI тогда
#         if not (CenterFace[0] in range(CenterROI[0] - 5, CenterROI[0] + 10) and CenterFace[1] in range(CenterROI[1] - 5,
#                                                                                                        CenterROI[
#                                                                                                            1] + 10)):
#     #       изменяем кроп кадра, высчитывая разность центров
#             xROI = xROI + (CenterFace[0] - CenterROI[0])
#             yROI = yROI + (CenterFace[1] - CenterROI[1])
#             yMaxROI = yMaxROI + (CenterFace[1] - CenterROI[1])
#             xMaxROI = xMaxROI + (CenterFace[0] - CenterROI[0])
#
#     # if xMaxROI > 640 or yMaxROI > 480:
#     #     ROI = frame[xROI:400, yROI:560] #?
#
#     # print(xROI, yROI, xROIpred, yROIpred)
#     if xROI < 0 or yROI<0 or xMaxROI > 640 or yMaxROI>480:
#         ROI = frame[0:480, 0:640]
#     else:
#         ROI = frame[yROI:yMaxROI, xROI:xMaxROI]
#
#     frame = cv2.rectangle(frame, (xROI, yROI), (xMaxROI, yMaxROI), (0, 0, 255), 3)
#     frame = cv2.circle(frame, CenterROI, 1, (244, 22, 244))
#
#     cv2.imshow("test", cv2.resize(frame, (640, 480)))
#     cv2.imshow("test1", cv2.resize(ROI, (640,480)))
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()


# Познакомимся с виртуальной камерой
# мы уже научились работать с нашей основной камерой но что если мы хотим вывести изображение
# допустим в google meet
# мы можем создать виртуальную камеру, которую потом сможем испольщавать как основную

# import cv2
# import pyvirtualcam
#
# with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#
#
#
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         cam.send(cv2.resize(frame, (640, 480)))
#         cam.sleep_until_next_frame()

# Задание сделать отслеживание лица и вывести на виртуальную камеру

# import cv2
# from cvzone.FaceDetectionModule import FaceDetector
# import pyvirtualcam
# yROI = 80
# yMaxROI = 400
# xROI = 80
# xMaxROI = 560
# #вычисляем центр лица, так как изначально мы не знаем координат лица задаим стандартное значение центра
# #нашего bboxа. Центр высчитывается как сумма координат x деленное на 2 и y деленное на  2
# CenterFace = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
# with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
#     cap = cv2.VideoCapture(0)
#
#     detector = FaceDetector(0.6)
#     while True:
#         ret, frame = cap.read()
#     #    метод copy берет копию кадра
#         ROI = frame.copy()
#     #    теперь высчитываем центр нашего прямоугольника "интереса"
#         CenterROI = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
#     #    теперь получаем данные об обнаруженном лице
#         frame, detectionData = detector.findFaces(frame, draw=False)
#         if len(detectionData) != 0:
#             x,y,w,h =  detectionData[0].get("bbox")
#             print(x,y,w,h)
#             CenterFace = [(x + x + w) // 2, (y + y + h) // 2]
#
#
#         #       Далее мы сделаем проверку если центр лица находится не в диапазоне центра ROI тогда
#             if not (CenterFace[0] in range(CenterROI[0] - 5, CenterROI[0] + 10) and CenterFace[1] in range(CenterROI[1] - 5,
#                                                                                                            CenterROI[
#                                                                                                                1] + 10)):
#         #       изменяем кроп кадра, высчитывая разность центров
#                 xROI = xROI + (CenterFace[0] - CenterROI[0])
#                 yROI = yROI + (CenterFace[1] - CenterROI[1])
#                 yMaxROI = yMaxROI + (CenterFace[1] - CenterROI[1])
#                 xMaxROI = xMaxROI + (CenterFace[0] - CenterROI[0])
#
#         # if xMaxROI > 640 or yMaxROI > 480:
#         #     ROI = frame[xROI:400, yROI:560] #?
#
#         # print(xROI, yROI, xROIpred, yROIpred)
#         if xROI < 0 or yROI<0 or xMaxROI > 640 or yMaxROI>480:
#             ROI = frame[0:480, 0:640]
#         else:
#             ROI = frame[yROI:yMaxROI, xROI:xMaxROI]
#
#         # frame = cv2.rectangle(frame, (xROI, yROI), (xMaxROI, yMaxROI), (0, 0, 255), 3)
#         # frame = cv2.circle(frame, CenterROI, 1, (244, 22, 244))
#
#         ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
#         cam.send(cv2.resize(ROI, (640, 480)))
#         cam.sleep_until_next_frame()
#







# import cv2
#
# from cvzone.HandTrackingModule import HandDetector
#
# cap = cv2.VideoCapture(0)
#
# detector = HandDetector(maxHands=1)
#
# while True:
#     ret, frame = cap.read()
#     frame = detector.findHands(frame)
#     lmList, bbox = detector.findPosition(frame)
#     if len(lmList) != 0:
#         # print("lm", lmList[8])
#         x1, y1 = lmList[8]
#         x2, y2 = lmList[12]
#         # print(x1, y1, x2, y2)
#         fingers = detector.fingersUp()
#         # print(fingers)
#         if fingers[1] == 1 and fingers[2] ==1:
#             length, img, lineInfo = detector.findDistance(8, 12, frame)
#             print(length)
#             # lineInfo =  x,y,w,h, centerX, centerY
#
#             # frame = cv2.circle(frame, (x1,y1), int(length), (0,255,255),-1)
#     # print(detector.fingersUp())
#
#
#
#     cv2.imshow("lesson_2", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break


####### ВСЕ ЧТО НИЖЕ ОБЪЯСНИТЬ!!!!!!!!!!!!!!!!!!!!
# Научимся рисовать при помощи пальца

import cv2

from cvzone.HandTrackingModule import HandDetector
import numpy as np
# cap = cv2.VideoCapture(0)
#
# detector = HandDetector(maxHands=1)
# xp, yp = 0,0
# imgCanvas = np.zeros((480, 640, 3), np.uint8)
# cv2.namedWindow("lesson_2")
# def nothing(x):
#     pass
# cv2.createTrackbar('R','lesson_2',0,255,nothing)
# cv2.createTrackbar('G','lesson_2',0,255,nothing)
# cv2.createTrackbar('B','lesson_2',0,255,nothing)
#
#
# while True:
#     r = cv2.getTrackbarPos('R', 'lesson_2')
#     g = cv2.getTrackbarPos('G','lesson_2')
#     b = cv2.getTrackbarPos('B','lesson_2')
#     ret, frame = cap.read()
#     frame = detector.findHands(frame)
#     lmList, bbox = detector.findPosition(frame)
#     if len(lmList) != 0:
#         # print("lm", lmList[8])
#         x1, y1 = lmList[8]
#         x2, y2 = lmList[12]
#         # print(x1, y1, x2, y2)
#         fingers = detector.fingersUp()
#         # print(fingers)
#         if fingers[1] == 1 and fingers[2] ==0:
#
#             # length, img, lineInfo = detector.findDistance(8, 12, frame)
#             # print(length)
#             print(f'before xp {xp},yp {yp} x1 {x1} y1 {y1}')
#             if xp == 0 and yp == 0:
#                 xp, yp = x1, y1
#
#             cv2.line(frame, (xp, yp), (x1, y1), (255,0,255), 25)
#             cv2.line(imgCanvas, (xp, yp), (x1, y1), (b,g,r), 25)
#             print(f' after xp {xp},yp {yp} x1 {x1} y1 {y1}')
#
#             xp, yp = x1, y1
#             print(f' afterxp {xp},yp {yp} x1 {x1} y1 {y1}')
#
#     imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
#     _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
#     imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
#     img = cv2.bitwise_and(frame, imgInv)
#     img = cv2.bitwise_or(img, imgCanvas)
#             # lineInfo =  x,y,w,h, centerX, centerY
#             # frame = cv2.circle(frame, (x1,y1), int(length), (0,255,255),-1)
#     # print(detector.fingersUp())
#
#
#
#     cv2.imshow("lesson_2", img)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
#
#


# теперь сделаем управление мышкой с помощью наших пальцев
# для этого импортируем библиотеку autopy
# import cv2
# import autopy
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# ##########################
# wCam, hCam = 640, 480
# frameR = 50  # Frame Reduction
# smoothening = 7
# #########################
#
# plocX, plocY = 0, 0
# clocX, clocY = 0, 0
#
# cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
# detector = HandDetector(maxHands=1)
# wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
#
# while True:
#     ret, frame = cap.read()
#     frame = detector.findHands(frame)
#     lmList, bbox = detector.findPosition(frame)
#     if len(lmList) != 0:
#         # print("lm", lmList[8])
#         x1, y1 = lmList[8]
#         x2, y2 = lmList[12]
#         # print(x1, y1, x2, y2)
#         fingers = detector.fingersUp()
#
#         cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
#         # print(fingers)
#         if fingers[1] == 1 and fingers[2] ==0:
#             # length, img, lineInfo = detector.findDistance(8, 12, frame)
#             # print(length)
#             # # lineInfo =  x,y,w,h, centerX, centerY
#             # print(lineInfo)
#             ######### НАЙТИ ОБЪЯСНЕНИЕ!!!!!!!!!!!!!!!!!!
#             x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
#             y3 = np.interp(y1, (frameR  , hCam - frameR), (0, hScr))
#
#             # 6. Smoothen Values
#             clocX = plocX + (x3 - plocX) / smoothening
#             clocY = plocY + (y3 - plocY) / smoothening
#             print(clocX,clocY)
#             # 7. Move Mouse
#             autopy.mouse.move(wScr - clocX, clocY)
#             cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
#
#             plocX, plocY = clocX, clocY
#
#             # frame = cv2.circle(frame, (x1,y1), int(length), (0,255,255),-1)
#     # print(detector.fingersUp())
#
#
#
#     cv2.imshow("lesson_2", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
#
















# работа с teachible machine
# import tensorflow.keras
# from PIL import Image, ImageOps
# import numpy as np
# import cv2
#
# cap  = cv2.VideoCapture(0)
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# classesFile = "labels.txt"
# classNames = []
# # открываем наш файл coco
# with open(classesFile, 'rt') as f:  # t - текстовый режим
#     classNames = f.read().rstrip('\n').split(
#         '\n')
# # Load the model
# model = tensorflow.keras.models.load_model('keras_model.h5')
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
# # Replace this with the path to your image
# # image = Image.open('test3.jpg')
#
# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# # image = ImageOps.fit(image, size, Image.ANTIALIAS)
#
# #turn the image into a numpy array
# #
#
# while True:
#     ret, frame = cap.read()
#     image = Image.fromarray(frame)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)
#     image_array = np.asarray(image)
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#
#     # Load the image into the array
#     data[0] = normalized_image_array
#
#     # run the inference
#     prediction = model.predict(data)
#
#     im_np = np.asarray(image)
#     cv2.putText(im_np, classNames[prediction[0].argmax()][1:], (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
#     im_np = cv2.resize(im_np,(480,480))
#     cv2.imshow("test", im_np)
#     keyCode = cv2.waitKey(1)
#
#     if cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) < 1:
#         break
