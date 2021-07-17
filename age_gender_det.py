# A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
detector = FaceDetector()

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
# задаем значения среднего вычетания
# нейронная сеть была обучена на множестве данных и все они были приведены к среднему значению по цвету
# поэтому, чтобы сеть работала корректно мы будем вычитать из нашего изображения цвета
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# в модели присутствует список значений возраста и пола
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Муж.', 'Жен.']

# Читаем модели для распознавания возраста и пола
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(0)
while True:
    hasFrame, frame = video.read()
    frameMinus = frame.copy()
    frameMinus[:,:,0] = frame[:,:,0] - 78.4263377603
    frameMinus[:, :, 1] = frame[:, :, 1] - 87.7689143744
    frameMinus[:, :, 2] = frame[:, :, 2] - 114.895847746

    if not hasFrame:
        break

    frame, faceBoxes = detector.findFaces(frame)
    print(faceBoxes)
    if len(faceBoxes) != 0:
        # face = faceBoxes[0].get("bbox")
        x, y, w, h = faceBoxes[0].get("bbox")
        # теперь возьмем конкретно лицо
        face = frame[y:y + h, x:x + w]

        cropframe = cv2.resize(face, (227, 227))
        # модели обучены на изображениях размером 227 на 227 пикселей. Брать всю картинку не имеет смысла
        # поэтому мы возьмем изменим размер нашего лица на 227х227 затем сделаем вычитание наших значений и не будем менять каналы
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        print(blob.shape)
        blobImage = blob.reshape(blob.shape[2],blob.shape[3],blob.shape[1])
        # теперь передаем наше обработанное изображение в нейронную сеть
        genderNet.setInput(blob)
        #Выполняет прямой проход для вычисления вывода слоя с именем b возвращает предсказение
        genderPreds = genderNet.forward()
        print(genderPreds)
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        # тоже самое проделываем и с определением возраста
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(frame, f'{gender}, {age}', (x+65, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2,
                    cv2.LINE_AA)

        # StacedBlobs = np.hstack([blobImage, cropframe])
        StackedResult = np.hstack([frame, frameMinus])
        cv2.imshow("blob", blobImage)
        cv2.imshow("crop", cropframe)
        cv2.imshow("Detecting age and gender", StackedResult)
        if cv2.waitKey(1) == ord("q"):
            break
