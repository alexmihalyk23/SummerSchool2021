#pip install opencv-python


import cv2
import numpy as np
#Для начала научимся считывать кадры с камеры

# cap = cv2.VideoCapture(0)
# также можно получать кадры и из видео
#cap = cv2.VideoCapture("путь_к_файлу")
# ret, frame = cap.read()
# cv2.imshow("lesson_1", frame)
# cv2.waitKey(0)

#также можно считывать изображение
# image = cv2.imread("test.png")
# cv2.imshow("image", image)
# cv2.waitKey(0)

# теперь будем постоянно опрашивать камеру
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     x,y,w,h = 200,200,100,100
#     #тут мы будем писать основной код
#     cv2.rectangle(frame, (x,y),(x+w,y+h), (200,22,222), -1)
#     cv2.circle(frame, (400,400),9,(200,255,22),-1)
#     cv2.line(frame,(0,0),(300,300),(200,22,22))
#
#     #Задание найти центр прямоугольника
#
#     cv2.circle(frame,((x+x+w)//2,(y+y+h)//2),i,(0,255,0),-1)
#
#
#
#     cv2.imshow("lesson_1", frame)
#     cv2.waitKey(1)

#работа с cap.get и  cap.set
# w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(w,h)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
# newW=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# newH=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(newW,newH)
# while True:
#     ret, frame = cap.read()
#     x,y,w,h = 200,200,100,100
#     #тут мы будем писать основной код
#     cv2.rectangle(frame, (x,y),(x+w,y+h), (200,22,222), -1)
#     cv2.circle(frame, (400,400),9,(200,255,22),-1)
#     cv2.line(frame,(0,0),(300,300),(200,22,22))
#
#     #Задание найти центр прямоугольника
#
#     cv2.circle(frame,((x+x+w)//2,(y+y+h)//2),2,(0,255,0),-1)
#
#
#
#     cv2.imshow("lesson_1", frame)
#     cv2.waitKey(1)
#Поработаем с трекбарами
# import numpy as np
#
# #для начала с помощью библиотеки numpy создадим изображение
# img = np.zeros((300,512,3), np.uint8)
# # #назовем наше окно как lesson_1
# cv2.namedWindow('lesson_1')
# # Теперь создадим трекбар для изменения цвета
# # первый параметр это название трекбара, второй - окно, на котором будет отображаться трекбар,
# #третий - начальное значение, четвертый - конечное значение, пятый - функция обратного вызова,
# # которая выполняется. каждый раз, когда значение трекбара изменяется
# # на данный момент мы создадим пустую функцию, которая будет делать "Ничего"
#
# def nothing(x):
#     pass
# cv2.createTrackbar('R','lesson_1',0,255,nothing)
# cv2.createTrackbar('G','lesson_1',0,255,nothing)
# cv2.createTrackbar('B','lesson_1',0,255,nothing)
# while(1):
#     cv2.imshow('lesson_1',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     # get current positions of four trackbars
#     r = cv2.getTrackbarPos('R','lesson_1')
#     g = cv2.getTrackbarPos('G','lesson_1')
#     b = cv2.getTrackbarPos('B','lesson_1')
#
#     img[:] = [b,g,r]
# cv2.destroyAllWindows()

#Теперь мы знаем как работать с трекбарами и поэтому мы можем поработать с картинками

# Задание, нарисовать на картинке круг который будет менять цвет, позицию и размер в реальном времени
#
# cv2.namedWindow('lesson_1')
# cap = cv2.VideoCapture(0)
# def nothing(x):
#     pass
# cv2.createTrackbar('R','lesson_1',0,255,nothing)
# cv2.createTrackbar('G','lesson_1',0,255,nothing)
# cv2.createTrackbar('B','lesson_1',0,255,nothing)
# cv2.createTrackbar('X', 'lesson_1', 0,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), nothing)
# cv2.createTrackbar('Y', 'lesson_1', 0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), nothing)
# cv2.createTrackbar('radius', 'lesson_1', 1,100,nothing)
# while(1):
#     ret, img = cap.read()
#
#
#
#     # get current positions of four trackbars
#     r = cv2.getTrackbarPos('R','lesson_1')
#     g = cv2.getTrackbarPos('G','lesson_1')
#     b = cv2.getTrackbarPos('B','lesson_1')
#     x = cv2.getTrackbarPos('X', 'lesson_1')
#     y = cv2.getTrackbarPos('Y', 'lesson_1')
#     radius = cv2.getTrackbarPos('radius', 'lesson_1')
#     cv2.circle(img, (x,y),radius,(b,g,r),-1)
#     cv2.imshow('lesson_1', img)
#
#
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()


#############

# img1 = cv2.imread('C:/Users/alexm/Pictures/english_1.png')
# img2 = cv2.imread('mask.png')
# img2 = cv2.resize(img2, (160,120))
# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols]
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# # threshold это пороговое значение
# # это значит что все пиксели, которые меньше порога становятся 0 а все что больше - 255
# # для того чтобы это работало корректно, сначала нам нужно перевести изображение в оттенки серого
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# print(mask)
# mask_inv = cv2.bitwise_not(mask)
# print(mask_inv)
# # !!!!! Дописать
# # Now black-out the area of logo in ROI
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# # Put logo in ROI and modify the main image
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# поработаем с нажатием мышки
# # получим кадр с камеры
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# # coordinates = np.zeros((1,2), np.int)
# #создадим массив coordinates
# coordinates = []
#создадим функцию, которая будет возвращать координаты где мы нажали на левую кнопку мыши
# def MouseClick(event, x, y, flags, params):
#     # если событие равно нажатию левой кнопкой мыши
#     if event == cv2.EVENT_LBUTTONDOWN:
#         #добавляем координаты x и y в наш массив
#         coordinates.append(x)
#         coordinates.append(y)
#         print(coordinates)
#
#
# while True:
#     print(len(coordinates))
#     # если координаты есть
#     if len(coordinates) !=0:
#         # нарисуем круг по нашим координатам
#         cv2.circle(img, (coordinates[0], coordinates[1]), 5, (0, 0, 200), -1)
#         # очистим наш массив, чтобы мы могли его снова заполнить
#         coordinates.clear()
#
#     cv2.imshow("lesson_1", img)
#     # и вызываем нашу функцию по нажатию мыши
#     cv2.setMouseCallback("lesson_1", MouseClick)
#     cv2.waitKey(1)


# теперь напишем кое-что интересное
# импортируем библиотеку numpy для работы с массивами
import numpy as np
# создадим массив 4 на 2
# заполним его нулями
circles = np.zeros((4,2), np.int)
# также создадим счетчик
counter = 0
# прочитаем кадр
cap = cv2.VideoCapture(0)
ret, img = cap.read()
# img = cv2.imread("test.png")
def MouseClick(event, x, y, flags, params):
    global counter
    # теперь мы добавим координаты в наш массив и прибавим 1 к счетчику
    if event == cv2.EVENT_LBUTTONDOWN:
        circles[counter] = x,y
        counter +=1
        print(circles)

while True:

    if counter == 4:
        width, height = 250,350
        # pts1 это координаты наших точек
        pts1 = np.float32([circles[0], circles[1], circles[2],circles[3]])
        # pts2  это координаты, куда будут помещены наши точки
        pts2 = np.float32([[0,0],[width,0], [0,height],[width,height]])
        print("pts",pts1,pts2)
        # теперь создадим матрицу matrix которая получит преобразование перспективы
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        # и теперь наконец деформируем наши точки по заданной выше маске и также задаем высоту и ширину
        imgOutput = cv2.warpPerspective(img, matrix, (width,height))
        cv2.imshow("output", imgOutput)

    for x in range(0,4):
        print(circles[x][0])
        cv2.circle(img,(circles[x][0],circles[x][1]),5, (0,0,200),-1)




    cv2.imshow("lesson_1", img)
    cv2.setMouseCallback("lesson_1", MouseClick)
    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite("img.png", imgOutput)
    elif key == ord("q"):
        break
    # cv2.waitKey(1)