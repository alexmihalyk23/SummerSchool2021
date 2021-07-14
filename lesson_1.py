# pip install opencv-python
# что такое изображение
#  по сути изображение это массив в котором лежит цвет кажлого пикселя
# по этому мы можем создать свое изображение

########################   1   ########################
import cv2
import numpy as np
# imgZ = np.ones((640, 480, 3))
#
# # # imgZ.
# #
# # imgZ[100][100][0] = np.random.randint(0, 255) / 255.0
# # imgZ[100][101][1] = np.random.randint(0, 255) / 255.0
# # imgZ[100][102][2] = np.random.randint(0, 255) / 255.0
# # imgZ[100][103][0] = np.random.randint(0, 255) / 255.0
# #uint8 Целые числа в диапазоне от 0 по   255 (числа размером 1 байт).
# imgZ = np.zeros((480,640,3), np.uint8)
# # cv2.circle(imgZ, (100,100), 5,(255,0,0), -1)
# # cv2.circle(imgZ, (200,10), 5,(0,255,0), -1)
# # cv2.circle(imgZ, (400,200), 5,(0,0,255), -1)
# # cv2.circle(imgZ, (100,200), 5,(255,0,255), -1)
# pts = np.array([[100,100],[200,10],[400,200],[100,200]], np.int32)
# #
# # #Это просто означает, что это неизвестное измерение, и мы хотим, чтобы numpy выяснил его.
# # # И numpy поймет это, посмотрев на "длину массива и оставшиеся размеры" и убедившись,
# # # что он удовлетворяет вышеупомянутым критериям
# pts = pts.reshape((-1,1,2))
# # print(pts)
# cv2.polylines(imgZ,[pts],True,(0,255,255))
# cv2.fillPoly(imgZ, [pts], (0,255,255))
# # cv2.polylines(imgZ, [pts], True, (255,0,0))
# cv2.imshow("s", imgZ)
# cv2.waitKey(0)
# colorized =

# imgZ.fill(0.69)
# Благодаря срезу мы можем менять только цвет, так как у нас массив состоит из
# 640 на 480 на 3
# где три это цвета RGB
# 0 - B 1 - G 2 - R
# imgZ[:,:,0]=255
# imgZ[:,:,1] = 0
# imgZ[:,:,2] = 0
# # imgZ[:,:,:] = 128
# print(imgZ[:,:,:])
# print(imgZ[:,:,0])
# # print(imgZ[:, :, 0])
# # # также мы можем выполнять все операции opencv над этим изображением
# # imgZ = cv2.resize(imgZ, (640,480))
# # # Рисование прямоугольника
# #
# # imgZ = cv2.rectangle(imgZ, (100, 40), (160, 100), (255, 0, 255), -1)
# # # рисование круга
# # imgZ = cv2.circle(imgZ, (130, 70), 3, (255, 0, 0), -1)
# # # написание текста на изображнии
# # imgZ = cv2.putText(imgZ,"hello",(0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
# # # стрелочка
# # imgZ = cv2.arrowedLine(imgZ,(20,40),(50,100),(200,0,0))

#
#
# # imgZ = cv2.drawMarker(imgZ, (200,200),(255,0,0), cv2.MARKER_DIAMOND)
# #
# # print(imgZ.shape)
# # print(imgZ)
# # print(imgZ[1][0])
# cv2.imshow("imgZ", imgZ)
# cv2.waitKey(0)

# Двоеточие в массиве это срез
# for i in range(0,600):
#     for j in range(0,400):
# теперь поподробнее про срез,
# мы можем выбрать в какой части массива(нашей картинке) будет какой-нибудь другой цвет
# import math
# x,y,w,h = 0,100,200,200
# imgZ[:,:,:] = 100
# # imgZ[100:200,10,2] = 255
#
# imgZ[y:y+h,x:x+w,1] = np.random.randint(0, 255)
# # imgZ[y+w,x,2] = 255
# # imgZ[:,:,1] = np.random.randint(0, 255)
# # imgZ[:,:,2] = np.random.randint(0, 255)
# img = cv2.imread("testF2.jpg")
# # print(img)
# # print(img[:,:,0])
# # imgZ = np.zeros([640, 480, 3])
# # imgZ.fill(0.69)
# imgZ = np.array(img[20:200,40:500])
# # cv2.circle()
# cv2.imshow("frame", imgZ)
# cv2.waitKey(0)

#
# import cv2
# import numpy as np
#
# imgZ = np.zeros((640, 480, 3))
# # imgZ.
# posX = 0
# posY = 40
# # Немного подробнее о waitkey
# while True:
#     key = cv2.waitKey(1)
#     if key != 27 and key != -1 and key != 8:
#         imgZ = cv2.putText(imgZ,str(chr(key)),(posX,posY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
#         posX+=15
#         if posX >= 480:
#             posX = 0
#             posY = posY + 40
#             imgZ = cv2.putText(imgZ, str(chr(key)), (posX, posY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
#
#     #
#     # print(imgZ.shape)
#     # print(imgZ)
#     # print(imgZ[1][0])
#     cv2.imshow("imgZ", imgZ)
#     if key == 27:
#         break
#


# Поработаем с трекбарами
import numpy as np
#
# #для начала с помощью библиотеки numpy создадим изображение
# img = np.zeros((480,640,3), np.uint8)
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
#
# print(img[:])
# while(1):
#     pts = np.array([[10,20],[30,45],[40,30],[20,20]], np.int32)
#     pts = pts.reshape((-1,1,2))
#     cv2.polylines(img, [pts], True, (255,0,222), 1)
#     cv2.imshow('lesson_1',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     # get current positions of four trackbars
#     r = cv2.getTrackbarPos('R','lesson_1')
#     g = cv2.getTrackbarPos('G','lesson_1')
#     b = cv2.getTrackbarPos('B','lesson_1')
#
#     img[:] = [b,g,r] # ОСТАНОВИЛИСЬ ТУТ
# cv2.destroyAllWindows()
#

# img = np.zeros([480,640,3], np.uint8)
# posX, posY = 320,100
# dx, dy = 1,1
# boxX, boxY = 560, 20
# counter = 0
# while True:
#
#     key = cv2.waitKey(1)
#     cv2.line(img, (320, 0), (320, 640), (255, 255, 255), 1)
#     cv2.ellipse(img, (posX,posY), (50,50), 20   , 0, 360, (255,0,255), -1)
#     posX +=dx
#     posY +=dy
#     # cv2.circle(img, (posX+50,posY), 4, (0,0,255), -1)
#     # cv2.circle(img, (boxX, boxY), 4, (0, 0, 255), -1)
#     # cv2.circle(img, (boxX, boxY+130), 4, (0, 0, 255), -1)
#     # if posX +50 == 640:
#     #     dx *=-dx
#     # if posY+50 == 480:
#     #     dy *=-dy
#     # if posX-50 == 0:
#     #     dx = -dx
#     # if posY-50 == 0:
#     #     dy = -dy
#
#
#     if posX +50 ==boxX  and posY in range(boxY-50, boxY+100) or posX +50 ==boxX and posY in range(boxY+100, boxY+130):
#         print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
#         dx = -dx
#
#     if posX-50 == 640:
#         counter+=1
#         posX, posY = 320,100
#
#     if posY+50 == 480:
#         dy *= -dy
#
#     if posY-50 == 0:
#         dy = -dy
#         # dx = -dx
#     if posX-120 == 0:
#         dx = -dx
#
#     cv2.rectangle(img, (boxX, boxY), (boxX+15, boxY+100), (255,255,255), -1)
#     if key == ord("w") and boxY !=0:
#         boxY -=20
#     if key == ord("s") and boxY+100 !=480:
#         boxY +=20
#     cv2.putText(img, str(counter), (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
#     # print(f"ellipseY: {posY}, ellipseX: {posX}, recY: {boxY}, recX: {boxX}")
#     cv2.imshow("test", img)
#     img.fill(0)
#
#     if key == ord("q"):
#         break



# print(imgZ)

# Для начала научимся считывать кадры с камеры

cap = cv2.VideoCapture(0)
# также можно получать кадры и из видео
# cap = cv2.VideoCapture("testF1.jpg")
# ret, frame = cap.read()
# x,y,w,h = 20,20,280,380
# # так как это массив чисел мы можем взять конкретную часть изображения
# crop = frame[y:y+h, x:x+w]
# #нарисуем прямоугольник на изображении для наглядности возьем те же
# # координаты что и для кропа
# frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(244,2,0))
#
# cv2.imshow("crop", crop)
# cv2.imshow("lesson_1", frame)
# cv2.waitKey(0)
#
# # также можно считывать изображение
# image = cv2.imread("test.png")
# cv2.imshow("image", image)
# cv2.waitKey(0)

#ОСТАНОВИЛИСЬ ТУ###################################

# crop = frame[y:y + h, x:x + w].copy()
#     cropGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     cropGray = cv2.cvtColor(cropGray, cv2.COLOR_GRAY2BGR)

# Cделать прямоугольник который будет двигаться и отскакивать от углов, и красить в серый
# x,y,w,h = 20,20,150,150
# dx,dy = 1,1
# while True:
#     ret, frame = cap.read()
#     if y+h == 480 or y == 0:
#         dy = -dy
#     if x+w == 640 or x == 0:
#         dx = -dx
#
#     # так как это массив чисел мы можем взять конкретную часть изображения
#
#
#
#     x+=dx
#     y+=dy
#     print(y+h)
#     crop = frame[y:y + h, x:x + w].copy()
#     cropGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     cropGray = cv2.cvtColor(cropGray, cv2.COLOR_GRAY2BGR)
#
#     frame[y:y+h, x:x+w] = cropGray
#     #нарисуем прямоугольник на изображении для наглядности возьем те же
#     # координаты что и для кропа
#     frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(244,2,0))
#
#     cv2.imshow("crop", crop)
#     cv2.imshow("lesson_1", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break

# Сделать отслеживание по цвету№№№№№№№№№№№№№№№№№№№№№


# ret, frame = cap.read()
# roi = cv2.selectROI("lesson_ 1",frame,True)
# cv2.imshow("lesson_1", frame)
# print(roi)
# x,y,w,h = roi
# r = 5
# rectX = (x - r)
# rectY = (y - r)
#
#
# dx,dy = 1,1
# while True:
#     ret, frame = cap.read()
#     if y+h == 480 or y == 0:
#         dy = -dy
#     if x+w == 640 or x == 0:
#         dx = -dx
#
#     # так как это массив чисел мы можем взять конкретную часть изображения
#
#
#
#     x+=dx
#     y+=dy
#     print(y+h)
#     crop = frame[y:y + h, x:x + w].copy()
#     # crop = frame[y:(y + 2 * r), x:(x + 2 * r)].copy()
#     cropGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     cropGray = cv2.cvtColor(cropGray, cv2.COLOR_GRAY2BGR)
#
#     frame[y:y+h, x:x+w] = cropGray
#     #нарисуем прямоугольник на изображении для наглядности возьем те же
#     # координаты что и для кропа
#     frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(244,2,0))
#
#     cv2.imshow("crop", crop)
#     cv2.imshow("lesson_1", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break


# ret, frame = cap.read()
# roi = cv2.selectROI("lesson_1",frame,True)
# cv2.imshow("lesson_1", frame)
# print(roi)
# x,y,w,h = roi
# r = 5
# rectX = (x - r)
# rectY = (y - r)
#
#
# dx,dy = 1,1
#
#
# while True:
#     ret, frame = cap.read()
#     if y+h == 480 or y == 0:
#         dy = -dy
#     if x+w == 640 or x == 0:
#         dx = -dx
#
#     # так как это массив чисел мы можем взять конкретную часть изображения
#
#
#
#     x+=dx
#     y+=dy
#     print(y+h)
#
#     crop = frame[y:y + h, x:x + w].copy()
#     crop = cv2.resize(crop, (640,480))
#     # crop = frame[y:(y + 2 * r), x:(x + 2 * r)].copy()
#     cropGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     cropGray = cv2.cvtColor(cropGray, cv2.COLOR_GRAY2BGR)
#     wI, hI, cI = frame[y:y + h, x:x + w].shape
#     print(f"w,h,c {wI}, {hI}")
#     cropGray = cv2.resize(cropGray,(hI,wI))
#
#     frame[y:y+h, x:x+w] = cropGray
#     #нарисуем прямоугольник на изображении для наглядности возьем те же
#     # координаты что и для кропа
#     frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(244,2,0))
#     cv2.imshow("crop", crop)
#     cv2.imshow("lesson_1", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break


# также можно считывать изображение



# теперь будем постоянно опрашивать камеру

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     x,y,w,h = 200,200,100,100
#     #тут мы будем писать основной код
#     cv2.rectangle(frame, (x,y),(x+w,y+h), (200,22,222), -1)
#     cv2.circle(frame, (400,400),9,(200,255,22),-1)
#     cv2.line(frame,(0,0),(300,300),(200,22,22))
#     crop = frame[y:y+h+200,x:x+w+100]
#
#     #Задание найти центр прямоугольника и нарисовать круг
#
#     cv2.circle(frame,((x+x+w)//2,(y+y+h)//2),1,(0,255,0),-1)
#     cv2.imshow("crop", crop)
#     cv2.imshow("lesson_1", frame)
#     cv2.waitKey(1)


# работа с cap.get и  cap.set
# cap = cv2.VideoCapture(0)
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

########################   4   ########################

# Теперь мы знаем как работать с трекбарами и поэтому мы можем поработать с картинками

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





####################НА 7 ЧИСЛО№№№№№№№№№№№№

# чтобы узнать какие события поддерживает opencv напишем такой код
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print( events )

# поработаем с нажатием мышки
# # получим кадр с камеры
# img = np.zeros((640,480,3))
# # coordinates = np.zeros((1,2), np.int)
# #создадим массив coordinates
# coordinates = []
# #создадим функцию, которая будет возвращать координаты где мы нажали на левую кнопку мыши
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

# # теперь мы знаем как отслеживать нажатие клавишы мыши и что такое трекбар
#
# # Задание, сделать аналог paint
#
# cv2.namedWindow('lesson_1')
# # cap = cv2.VideoCapture(0)
# # img = cap.read[1]
#
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # coordinates = np.zeros((1,2), np.int)
# #создадим массив coordinates
# coordinates = []
# drawing = False
# r,g,b = 0,0,0
# radius = 0
# #создадим функцию, которая будет возвращать координаты где мы нажали на левую кнопку мыши
# def MouseClick(event, x, y, flags, params):
#     global drawing, r,g,b, radius, color
#     # если событие равно нажатию левой кнопкой мыши
#     if event == cv2.EVENT_LBUTTONDOWN:
#         #добавляем координаты x и y в наш массив
#         drawing = True
#         coordinates.append(x)
#         coordinates.append(y)
#         print(coordinates)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if cleaning == False:
#                 cv2.circle(img, (x,y), radius,(b,g,r),-1)
#             else:
#                 cv2.circle(img, (x, y), radius, (0, 0, 0), -1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#
# def nothing(x):
#     pass
# cv2.createTrackbar('R','lesson_1',0,255,nothing)
# cv2.createTrackbar('G','lesson_1',0,255,nothing)
# cv2.createTrackbar('B','lesson_1',0,255,nothing)
# cv2.createTrackbar('Radius','lesson_1',0,100,nothing)
# cv2.createTrackbar('cleaning','lesson_1',False,True,nothing)
# # cap = cv2.VideoCapture(0)
# # ret, img = cap.read()
# while True:
#
#     print(len(coordinates))
#     # если координаты есть
#     r = cv2.getTrackbarPos('R', 'lesson_1')
#     g = cv2.getTrackbarPos('G','lesson_1')
#     b = cv2.getTrackbarPos('B','lesson_1')
#     radius = cv2.getTrackbarPos('Radius', 'lesson_1')
#     cleaning = cv2.getTrackbarPos('cleaning', 'lesson_1')
#     if len(coordinates) !=0:
#         # нарисуем круг по нашим координатам
#         cv2.circle(img, (coordinates[0], coordinates[1]), radius, (b, g,r), -1)
#         # очистим наш массив, чтобы мы могли его снова заполнить
#         coordinates.clear()
#
#     cv2.imshow("lesson_1", img)
#     # и вызываем нашу функцию по нажатию мыши
#     cv2.setMouseCallback("lesson_1", MouseClick)
#     cv2.waitKey(1)

# # Доп задание сделать рисование по зажатой клавише мыши
#
# def MouseClick(event, x, y, flags, params):
#     # если событие равно нажатию левой кнопкой мыши
#     if event == cv2.EVENT_LBUTTONDOWN:
#         #добавляем координаты x и y в наш массив
#         coordinates.append(x)
#         coordinates.append(y)
#         print(coordinates)
#
# def nothing(x):
#     pass
# cv2.createTrackbar('R','lesson_1',0,255,nothing)
# cv2.createTrackbar('G','lesson_1',0,255,nothing)
# cv2.createTrackbar('B','lesson_1',0,255,nothing)
# cv2.createTrackbar('Radius','lesson_1',0,100,nothing)
#
#
# while True:
#     print(len(coordinates))
#     # если координаты есть
#     r = cv2.getTrackbarPos('R', 'lesson_1')
#     g = cv2.getTrackbarPos('G','lesson_1')
#     b = cv2.getTrackbarPos('B','lesson_1')
#     radius = cv2.getTrackbarPos('Radius', 'lesson_1')
#     if len(coordinates) !=0:
#         # нарисуем круг по нашим координатам
#         cv2.circle(img, (coordinates[0], coordinates[1]), radius, (b, g,r), -1)
#         # очистим наш массив, чтобы мы могли его снова заполнить
#         coordinates.clear()
#
#     cv2.imshow("lesson_1", img)
#     # и вызываем нашу функцию по нажатию мыши
#     cv2.setMouseCallback("lesson_1", MouseClick)
#     cv2.waitKey(1)

#### ВСЕ ВЫШЕ СДЕЛАНО #######################

#################НА 08 #############


#
# img1 = cv2.imread("JOJO.jpg")
# img1 = cv2.resize(img1, (720,1200))
# img2 = cv2.imread("testF2.jpg")
# img3 = cv2.flip(img1, 180)
#
# result = img1+img3
#
# waighted = cv2.addWeighted(img1, 1.0,img3, 0.6,0)
#
# rows, cols, channels = img2.shape
# roi = img1[0:rows, 0:cols]
#
# img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2Gray, 250,255,cv2.THRESH_BINARY_INV)
#
# mask_inv = cv2.bitwise_not(mask)
#
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows,0:cols] = dst
#
# cv2.imshow("test", img1)
#
# cv2.imshow("mask", mask)
# res = cv2.Canny(img3, 100,200)
# ## cv2.add (155,211,79) + (50,170,200) = 205,389,279 конвертнется в (205,255,255)
# # ret,mask = cv2.threshold(img1, 100,255, cv2.THRESH_BINARY_INV)
# # mask_inv = cv2.bitwise_not(img1,img1, mask=mask)
#
# # img2[0:rows,0:cols]  = dst
#
#
# rows, cols,channels = img1.shape
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# waighted = cv2.resize(waighted, (480,640))
# # cv2.imshow("img1",img1)
# # cv2.imshow("img2",img2)
# # cv2.imshow("img3",img3)
# # cv2.imshow("waighted",waighted)
# cv2.imshow("result",res)
# cv2.waitKey(0)
#
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     # Любые края с градиентом интенсивности больше maxVal обязательно будут краями, а те, что ниже minVal, обязательно
#     #не будут краями, поэтому отбрасывайте.Те, кто находится между этими двумя порогами, классифицируются как
#     # ребра или неребра в зависимости от их связности.Если они соединены с пикселями с "четкими краями", они
#     # считаются частью краев.В противном случае они также выбрасываются.
#     edges = cv2.Canny(frame, 50,150)
#     cv2.imshow("img", edges)
#     if cv2.waitKey(1) == ord("q"):
#         break







# cv2.namedWindow("lesson_08")
#
# def nothing(x):
#     pass
#
# cap = cv2.VideoCapture(0)
# cv2.createTrackbar("HUE min", "lesson_08",0, 179, nothing)
# cv2.createTrackbar("HUE max", "lesson_08",179, 179, nothing)
# cv2.createTrackbar("SAT min", "lesson_08",0, 255, nothing)
# cv2.createTrackbar("SAT max", "lesson_08",255, 255, nothing)
# cv2.createTrackbar("VALUE min", "lesson_08",0, 255, nothing)
# cv2.createTrackbar("VALUE max", "lesson_08",255, 255, nothing)
#
#
# while True:
#     ret, img = cap.read()
#     # img = cv2.resize(img, (0,240))
#
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     h_min = cv2.getTrackbarPos("HUE min", "lesson_08")
#     h_max = cv2.getTrackbarPos("HUE max", "lesson_08")
#     s_min = cv2.getTrackbarPos("SAT min", "lesson_08")
#     s_max = cv2.getTrackbarPos("SAT max", "lesson_08")
#     v_min = cv2.getTrackbarPos("VALUE min", "lesson_08")
#     v_max = cv2.getTrackbarPos("VALUE max", "lesson_08")
#
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#
#     mask = cv2.inRange(imgHSV, lower, upper)
#     result = cv2.bitwise_and(img, img, mask=mask)
#     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     stackedImg = np.hstack([img,result])
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
#
#     # cv2.imshow("original", img)
#     # # cv2.imshow("HSV", imgHSV)
#     cv2.imshow("mask", opening)
#     cv2.imshow("result", stackedImg)
#     if cv2.waitKey(1) == ord("q"):
#         break
#
################# Закончили тут№№№№№№№№№№№№№№№№
# переходим в файл for_npy.py

# # создадим переменную load from disk  и устанавливаем на True
#
#
# # Эта переменная определяет, хотим ли мы загрузить диапазон цветов из #memory или использовать те, которые определены здесь.
# hsv_value = np.load('hsv_value.npy')
# print(hsv_value)
# lower_range = hsv_value[0]
# upper_range = hsv_value[1]
# kernel = np.ones((5,5), np.uint8)
# # # Инициализация холста, на котором мы будем рисовать
# canvas = None
# # Порог шума
# noiseth = 800
# cap = cv2.VideoCapture(0)
# x1,y1 = 0,0
# while True:
#     ret, frame = cap.read()
#     rgbimg = frame.copy()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#     # erode это яркие области изображения становятся тоньше, а темные - больше.
#     # dilate Яркая область нашей маски расширяется вокруг черных областей фона.
#     mask = cv2.erode(mask,kernel,iterations = 1)
#     mask = cv2.dilate(mask, kernel, iterations=2)
#     # mask1 = cv2.erode(mask1, kernel,iterations=2)
#     # cv2.imshow("mask Erode", mask1)
#
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # for i in range(len(contours)):
#     cnt = contours[0]
#     # contourArea это площадь контура
#     area = cv2.contourArea(cnt)
#     print(area)
#     cv2.drawContours(rgbimg, contours, 0, (0), 5)
#     # Убедитесь, что контур присутствует и его размер больше
#     # порога шума.
#     if canvas is None:
#     # задаем черный холст с таким же размером как и наше изображние для этого используем zeros_like
#         canvas = np.zeros_like(rgbimg)
#     ## если контуры есть и их площадь больше шума
#     if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > 800:
#         # берем максимальный размер контура
#         c = max(contours, key = cv2.contourArea)
#         print(c)
#         x2,y2,w,h = cv2.boundingRect(c)
#         cv2.rectangle(rgbimg, (x2,y2), (x2+w,y2+h), (0), 5)
#         if x1 == 0 and y1 == 0:
#             x1,y1= x2,y2
#         else:
#             canvas = cv2.line(canvas, (x1, y1), (x2, y2), [128, 128, 0], 4)
#         x1,y1= x2,y2
#     else:
#         x1,y1 =0,0
#     masked = cv2.bitwise_and(rgbimg,rgbimg,mask=mask)
#     rgbimg = cv2.add(rgbimg, canvas)
#     stack = np.hstack([rgbimg, canvas])
#     cv2.imshow("tet", rgbimg)
#     cv2.imshow("bg", mask)
#     cv2.imshow("stack", stack)
#     cv2.imshow("masked", masked)
#     cv2.waitKey(1)


# вспомним как мы создаем маски

import cv2

# cap = cv2.VideoCapture(0)
#
# face_cascade = cv2.CascadeClassifier("haar.xml")
#
# while True:
#     ret, frame =  cap.read()
#     mask = np.zeros(frame.shape[:2], np.uint8)
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=10)
#     print(faces)
#     points = []
#     for x, y, w, h in faces:
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
#         points.append([x,y,x+w,y+h])
#         cv2.rectangle(mask, (x,y), (x+w, y+h), (255), -1)
#
#         masked = cv2.bitwise_and(frame, frame, mask=mask)
#         print(mask.shape)
#         # Задание, заблюрить обнаруженное лицо, воспользоваться способом объединения двух фотографий
#         cv2.imshow("masked", mask)
#         cv2.imshow("masked1", masked)
#     cv2.imshow("lesson_2", frame)
#
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()

# import cv2
#
# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier("haar.xml")
#
# while True:
#     ret, frame =  cap.read()
#     mask = np.zeros(frame.shape[:2], np.uint8)
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=10)
#     print(faces)
#     points = []
#     for x, y, w, h in faces:
#         # frame_bg = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255))
#         points.append([x,y,x+w,y+h])
#         cv2.rectangle(mask, (x,y), (x+w, y+h), (255), -1)
#         # frame_bg[y:y+h, x:x+h].fill(0)
#
#         masked = cv2.bitwise_and(frame, frame, mask=mask)
#         masked1 = cv2.blur(masked, (55,55))
#         masked1 = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
#         masked1 = cv2.cvtColor(masked1, cv2.COLOR_GRAY2BGR)
#
#         frame[y:y+h, x:x+h] = masked1[y:y+h, x:x+h]
#         print(mask.shape)
#         # Задание, заблюрить обнаруженное лицо, воспользоваться способом объединения двух фотографий
#         cv2.imshow("masked", mask)
#         cv2.imshow("masked1", masked)
#         # cv2.imshow("frame1", frame_bg)
#     cv2.imshow("lesson_2", frame)
#
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()


########################   7   ########################

# теперь напишем кое-что интересное
# импортируем библиотеку numpy для работы с массивами
# import numpy as np
# # создадим массив 4 на 2
# # заполним его нулями
# circles = np.zeros((4,2), np.int)
# # также создадим счетчик
# counter = 0
# # прочитаем кадр
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
#
# # img = cv2.imread("test.png")
# def MouseClick(event, x, y, flags, params):
#     global counter
#     # теперь мы добавим координаты в наш массив и прибавим 1 к счетчику
#     if event == cv2.EVENT_LBUTTONDOWN:
#         circles[counter] = x,y
#         counter +=1
#         print(circles)
#
# while True:
#
#     if counter == 4:
#         width, height = 250,350
#         # pts1 это координаты наших точек
#         pts1 = np.float32([circles[0], circles[1], circles[2],circles[3]])
#         # pts2  это координаты, куда будут помещены наши точки
#         pts2 = np.float32([[0,0],[width,0], [0,height],[width,height]])
#         print("pts",pts1,pts2)
#         # теперь создадим матрицу matrix которая получит преобразование перспективы
#         matrix = cv2.getPerspectiveTransform(pts1,pts2)
#         # и теперь наконец деформируем наши точки по заданной выше маске и также задаем высоту и ширину
#         imgOutput = cv2.warpPerspective(img, matrix, (width,height))
#         cv2.imshow("output", imgOutput)
#
#     for x in range(0,4):
#         print(circles[x][0])
#         cv2.circle(img,(circles[x][0],circles[x][1]),5, (0,0,200),-1)
#
#
#
#
#     cv2.imshow("lesson_1", img)
#     cv2.setMouseCallback("lesson_1", MouseClick)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         cv2.imwrite("img.png", imgOutput)
#     elif key == ord("q"):
#         break


########################3на 13 чис=ло№№№№№№№№№№№№№№№№№№№№№№№
# Features Detector
# Как вы собираете пазлы?
# https://docs.opencv.org/3.4/df/d54/tutorial_py_features_meaning.html


# Для начала импортируем необходимые библиотеки и прочитаем два изображения

########################   9   ########################

# теперь благодаря пролученным знаниям мы сделаем дополненную реальность

import cv2
import numpy as np
import cvzone

imgTarget = cv2.imread("testF1.jpg")
cap = cv2.VideoCapture(0)
Video = cv2.VideoCapture("C:/Users/alexm/Desktop/smart_drones.mp4")

detection = False
frameCounter = 0

imgVideo = Video.read()[1]
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1,None)
img2 = np.zeros([wT, hT])
# imgWarp = np.zeros([wT,hT])
while True:

    ret, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    if detection == False:
        Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == Video.get(cv2.CAP_PROP_FRAME_COUNT):
            Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        ret, imgVideo = Video.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
    if len(good) > 45:
        detection = True
        # ОБЪЯСНИТЬ
        # https://docs.opencv.org/4.5.2/db/d27/tutorial_py_table_of_contents_feature2d.html
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
          #рассказать https://docs.opencv.org/4.5.2/d1/de0/tutorial_py_feature_homography.html
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        # StackedImages = cvzone.stackImages([imgWebcam,imgWarp,imgTarget, imgFeatures,imgWarp,imgAug],3, 0.5)

    cv2.imshow("features", imgAug)
    cv2.imshow("img2", img2)
    cv2.imshow("imgWarp", imgFeatures)
    # cv2.imshow("maskNew", imgWarp)
    # cv2.imshow("target", imgTarget)
    # cv2.imshow("webcam", imgWebcam)
    # cv2.imshow("video", imgVideo)

    cv2.waitKey(1)
    frameCounter += 1
