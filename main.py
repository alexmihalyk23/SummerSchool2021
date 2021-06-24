#Импортируем необходимые библиотеки opencv-pytohn и pyvirtualcam

import pyvirtualcam

import cv2
# скачиваем каскад хаара
face_cascade = cv2.CascadeClassifier("haar.xml")
#добавим заранее просчитнные координаты
yROI = 80
yMaxROI = 400
xROI = 80
xMaxROI = 560
# вычисляем центр лица, так как изначально мы не знаем координат лица задаим стандартное значение центра
# нашего bboxа. Центр высчитывается как сумма координат x деленное на 2 и y деленное на  2
CenterFace = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
#теперь создаем виртуальную камеру
with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    # и получаем данные с реальной камеры
    cap = cv2.VideoCapture(0)
    print(f'Using virtual camera: {cam.device}')
    # frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        #теперь читаем кадр
        ret, frame = cap.read()
        # метод copy берет копию кадра
        ROI = frame.copy()
        # теперь высчитываем центр нашего прямоугольника "интереса"
        CenterROI = [(xROI + xMaxROI) // 2, (yROI + yMaxROI) // 2]
        # теперь получаем данные об обнаруженном лице
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
        # в нашей переменной теперь хранятся хначения координат лица
        for x, y, w, h in faces:
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # высчитываем центр лица
            CenterFace = [(x + x + w) // 2, (y + y + h) // 2]
            # frame = cv2.circle(frame, CenterFace, 1, (244, 22, 244))
            #Далее мы сделаем проверку если центр лица находится не в диапазоне центра ROI тогда
        if not (CenterFace[0] in range(CenterROI[0] - 5, CenterROI[0] + 10) and CenterFace[1] in range(CenterROI[1] - 5,
                                                                                                       CenterROI[
                                                                                                        1] + 10)):
            # изменяем кроп кадра, высчитывая разность центров
            xROI = xROI + (CenterFace[0] - CenterROI[0])
            yROI = yROI + (CenterFace[1] - CenterROI[1])
            yMaxROI = yMaxROI + (CenterFace[1] - CenterROI[1])
            xMaxROI = xMaxROI + (CenterFace[0] - CenterROI[0])

        if xROI < 0 or yROI < 0 or xMaxROI > 640 or yMaxROI > 480:
            ROI = frame[0:480, 0:640]
        else:
            ROI = frame[yROI:yMaxROI, xROI:xMaxROI]

        cv2.waitKey(0)
        frame = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        cam.send(cv2.resize(frame,(640,480)))
        cam.sleep_until_next_frame()