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
from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
#инициализируем FaceDetector, который принимает параметр с какой точностью будет распознаваться лицо
detector = FaceDetector(0.6)
while True:
    ret, frame = cap.read()
    # теперь вызовим метод findFaces, передадим параметры изображение и зададим отрисовку
    frame, detectionData = detector.findFaces(frame, draw=False)
    # нам вернут уже обработанный кадр и данные о лице, где будет наше лицо
    print(detectionData)
    if len(detectionData) !=0:
        frame = cv2.circle(frame, detectionData[0].get("center"), 10,(0,0,255),-1)
        frame = cv2.rectangle(frame, detectionData[0].get("bbox"), (255,0,0))
        print(detectionData[0].get("score")[0])
        print(detectionData[0].get("bbox")[0])
        cv2.putText(frame, str(f'{int(detectionData[0].get("score")[0] * 100)}%'), (detectionData[0].get("bbox")[0],detectionData[0].get("bbox")[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2)

    cv2.imshow("lesson_2", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break



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
#
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
#     cv2.imshow("lesson_2", StackedImages)
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()
#




# import cv2
# from cvzone.HandTrackingModule import HandDetector
#
# cap = cv2.VideoCapture(0)
#
# detector = HandDetector(maxHands=2)
#
# while True:
#     ret, frame = cap.read()
#     frame = detector.findHands(frame)
#
#
#
#     cv2.imshow("lesson_2", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
