import cv2
import numpy as np
# https://www.murtazahassan.com/yolo-v3-using-opencv-p-2/
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("C:\\Users\\alexm\\Videos\\VID_20190919_204244.mp4")
whT = 256
confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
# открываем наш файл coco
with open(classesFile, 'rt') as f:  # t - текстовый режим
    classNames = f.read().rstrip('\n').split(
        '\n')
print(classNames)
# rstip удаление ненужных переносов строкс
 # а split разделение слов через переход строки

# # выведем все классы на печать
# # print(classNames)

# # файлы модели
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3-tiny.weights"
# Darknet - это инфраструктура нейронных сетей
# с открытым исходным кодом, написанная на C и CUDA.
# Он быстрый, простой в установке и поддерживает вычисления CPU и GPU.
# Читает модель сети, хранящуюся в файлах модели Darknet
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# так как мы используем библиотеку opencv-python
# то в качестве серверной части будем использовать opencv
# opencv это наш backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# работа на процессоре
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Итак, теперь, когда у нас есть массивы,
# которые содержат всю информацию о блоках,
# # мы можем отфильтровать блоки с низким уровнем
# # достоверности и создать список соответствующих блоков,
# # содержащих объекты.
# # Мы создадим новую функцию с именем findObjects.
# #
def findObjects(outputs, img):
    hT, wT, cT = img.shape  # shape возвращает, высоту, ширину и цвет
    bbox = []  # Bounding box Наш прямоугольник, который будет рисоваться на объекте
    # также создадим два списка Id класса и значением достоверности самого высокого класса
    classIds = []
    confs = []
    # Теперь мы пройдемся по 3 выходам
    # и получим поля один за другим.
    # Мы будем называть каждый блок det,
    # сокращенным для обнаружения,
    # так как он содержит больше информации,
    # чем просто координаты блока.
    for output in outputs:
        # мы получаем (картинка) yolo.png
        # print(output)
        # теперь пройдемся по всем элементам в output
        for det in output:
            # теперь возьмем все данные кроме первых 5-ти
            scores = det[5:]

            # print(scores)
            # и теперь берем максимальное значение
            classId = np.argmax(scores)
            # теперь в переменную confedence передаем наиболее вероятный элемент в scores
            confidence = scores[classId]
            # если confidence больше нашего значения
            #               порог доверия
            if confidence > confThreshold:
                # print(confidence)
                # det 0 1 2 3 0- центр по x в процентах 1- центр по y в процентах 2- центр по w в процентах 3- центр по h в процентах
                # print(det[2])
                # соответственно чтобы получить ширину и высоту, мы можем просто умножить
                w, h = int(det[2] * wT), int(det[3] * hT)
                # чтобы узнать координаты по x и y воспользуемся формулой
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                # добавим наши значения в bbox
                bbox.append([x, y, w, h])
                # добавим Id найденных классов в classIds
                classIds.append(classId)
                # добавим уверенноссть в conf
                confs.append(float(confidence))
                # print(confs)
                # теперь мы уже можем нарисовать квадрат и вывести на экран
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    # но может возникнуть проблема когда обнаруживается один и тот же объект,
    # из-за чего на одном объекте может быть несколько квадратов
    # Чтобы этого избежать, мы будем использовать Non Max Suppression.
    # Проще говоря, NMS устраняет перекрывающиеся поля.
    # Он находит перекрывающиеся блоки и затем,
    # основываясь на их уверенности, выбирает блок максимальной достоверности
    # и подавляет все не максимальные блоки.
    # Поэтому мы будем использовать встроенную функцию NMSBoxes.
    # Мы должны ввести ограничивающие точки, их значения достоверности, доверительный порог
    # и nmsThreshold. Функция возвращает индексы после исключения.
    # nmsThreshold это порог. Чем ниже порог тем агрессивнее работает nms
    # будет меньше повтряющихся прямоугольников на обном объекте

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        print(indices)
        # мы видим что у нас есть еще одна скобка
        # чтобы ее убрать просто напишем так
        i = i[0]
        # print(i)
        # print(bbox)
        box = bbox[i]
        # print(classNames)
        # print(classIds[i]) # номер нашего класса
        # print(classNames[67])
        # print(classNames[classIds[i]])
        # print(box)
        # теперь получаем координаты нашего box-а
        x, y, w, h = box[0], box[1], box[2], box[3]
        print(x,y,w,h)
        # print(x,y,w,h)
        # теперь рисуем прямоугольник
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # и пишем название класса и процент уверенности
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    # read передает два аргумента True если кадр есть и само изображение
    success, img = cap.read()
    timer = cv2.getTickCount()
    # Наша сеть принимает только определенный тип изображения,
    #  поэтому мы поменяем формат входного кадра формат
    # двоичных данных (blob) Мы 4 - мерный блоб из изображения.
# Опционально изменяет размеры и обрезает изображение по
    # центру, вычитает средние значения,
# масштабирует значения по коэффициенту масштабирования,
# меняет синий и
    # красный каналы.Мы будем хранить все значения по умолчанию.
 # Теперь, основываясь на том, какой размер изображения
    # мы использовали при загрузке файлов  cfg и weight,
    # мы установим размер изображения.
    # Поскольку мы использовали 320,
    # мы установим наш параметр whT (widthHeightTarget) на 320.
    # если вы используете 256 или какие либо другие не забудьте поменять в конфиге
    #           можем изменять вычитание цветов, мы оставим их по стандарту [0,0,0] 1 это swapRB
    # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    # теперь отправим наше blob изображение в нейросеть
    net.setInput(blob)
    # загугли входные слои нейросети
    layersNames = net.getLayerNames()
    # это все наши слои
    # print(layersNames)
    # теперь отбразим только выходные слои (те которые не связаны)
    # print(net.getUnconnectedOutLayers())
 # Это возвращает все имена, но нам нужны только имена
 # выходных слоев. Таким образом, мы можем использовать
 # функцию getUnconnectedOutLayers,
 #  которая возвращает индексы выходных слоев.
 #   Теперь мы можем просто использовать эти индексы,
 #    чтобы найти имена из нашего списка layerNames.
 #    Поскольку мы используем 0 в качестве первого
 #     элемента, мы должны вычесть 1 из индексов,
 #     поэтому получаем функцию getUnconnectedOutLayers.
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # Теперь мы можем запустить прямой проход и найти выходы сети.
    outputs = net.forward(outputNames)

    # вызываем нашу функцию и передаем наш outputs и img
    findObjects(outputs, img)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 128), 2)
    if int(fps) <= 10:
        cv2.putText(img, "FPS", (30,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    else:
        cv2.putText(img, "FPS", (30,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()

