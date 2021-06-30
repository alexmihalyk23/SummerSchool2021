# pip install opencv-python

########################   1   ########################
import cv2
import numpy as np
# Для начала научимся считывать кадры с камеры

# cap = cv2.VideoCapture(0)
# также можно получать кадры и из видео
# cap = cv2.VideoCapture("путь_к_файлу")
# ret, frame = cap.read()
# cv2.imshow("lesson_1", frame)
# cv2.waitKey(0)

# также можно считывать изображение
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

# также мы и сами можем создать изображение, ведь изображение это всего лишь массив из чисел
# import numpy as np
#
# imgZ = np.zeros((640,480,3))
# # imgZ.
# imgZ.fill(0.69)
# imgZ[100][100][0] = np.random.randint(0,255)/255.0
# imgZ[100][101][1] = np.random.randint(0,255)/255.0
# imgZ[100][102][2] = np.random.randint(0,255)/255.0
# imgZ[100][103][0] = np.random.randint(0,255)/255.0
# for i in range(40,100):
#     for j in range(40,100):
#         imgZ[i][j][0] = 128
#
#         imgZ[i][j][1] = 20
#
#         imgZ[i][j][2] =200
#
# # также мы можем выполнять все операции opencv над этим изображением
# # imgZ = cv2.resize(imgZ, (640,480))
# imgZ = cv2.rectangle(imgZ, (100,40),(160,100),(255,0,255), -1)
#
# print(imgZ.shape)
# print(imgZ)
# print(imgZ[1][0])
# cv2.imshow("imgZ", imgZ)
# cv2.waitKey(0)
# print(imgZ)

########################   2   ########################

# работа с cap.get и  cap.set
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

########################   3   ########################

# Поработаем с трекбарами
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

########################   5   ########################

#############
# Наложение одной картинки на другую
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

########################   6   ########################

# поработаем с нажатием мышки
# # получим кадр с камеры
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# # coordinates = np.zeros((1,2), np.int)
# #создадим массив coordinates
# coordinates = []
# создадим функцию, которая будет возвращать координаты где мы нажали на левую кнопку мыши
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

########################   8   ########################

# Features Detector
# Как вы собираете пазлы?
# https://docs.opencv.org/3.4/df/d54/tutorial_py_features_meaning.html


# Для начала импортируем необходимые библиотеки и прочитаем два изображения
import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
#
# img1 = cv2.imread('testF1.jpg')
#
# img2 = cv2.imread('testF2.jpg')
#
# # инициализируем наш детектор orb для нас он подойдет лучше всего, с остальными можно ознакомиться  в документации
# orb = cv2.ORB_create(nfeatures=1000)
# while True:
#     imgweb = cap.read()[1]
# # keypoints наши ключевые точки на изображении
# # descriptor =
# # первый параметр это наше изображение, второй это маска по которой будут определятся точки
# # в нашем случае его у нас нет
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     # kp2, des2 = orb.detectAndCompute(img2, None)
#     kp2, des2 = orb.detectAndCompute(imgweb, None)
#     # теперь для наглядности отобразим наши точки на изображении
#     imgkp1 = cv2.drawKeypoints(img1, kp1,None)
#     imgkp2 = cv2.drawKeypoints(imgweb, kp2,None)
#
#     #Что за дескрипторы
#     #Дескрипторы находят фичи кадра, то есть ищет признаки на кадре
#     print(des1.shape)
#     print(des1[0])
#     # тут мы используем метод грубого перебора для нахождения признаков на обоих изображениях
#     #
#     bf = cv2.BFMatcher()
#     # так как у нас два изображения то 3 параметорм задаем 2
#     matches = bf.knnMatch(des1,des2,2)
#     # теперь найдем хорошие признаки, которые есть у обоих изображениях
#     good = []
#     for m,n in matches:
#         # если дистанция первого кадра меньше 75% второго, то относим к хорошим
#         # УТОЧНИТЬ
#         if m.distance < 0.75 * n.distance:
#             good.append([m])
#         # print(f"m - {m.distance}, n - {n.distance}")
#     print(len(good))
#     # если хороших признаков больше 30, тогда мы можем предположить что кадр обнаружен
#     if len(good) > 30:
#         cv2.putText(imgweb, "blagodarnost", (50,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255))
#     # для наглядности отобразим найденные признаки на обоих изображениях
#     img3 = cv2.drawMatchesKnn(img1,kp1,imgweb,kp2,good,None)
#
#     cv2.imshow("img1", img1)
#     cv2.imshow("imgkp1", imgkp1)
#     cv2.imshow("imgkp2", imgweb)
#     cv2.imshow("img2", img2)
#     cv2.imshow("img3", img3)
#
#     cv2.waitKey(1)

########################   9   ########################

# теперь благодаря пролученным знаниям мы сделаем дополненную реальность

# import cv2
# import numpy as np
# import cvzone
#
# imgTarget = cv2.imread("C:/users/alexm/Pictures/testAR3.jpg")
# cap = cv2.VideoCapture(0)
# Video = cv2.VideoCapture("C:/Users/alexm/Pictures/testAR2.mp4")
#
# detection = False
# frameCounter = 0
#
# imgVideo = Video.read()[1]
# hT, wT, cT = imgTarget.shape
# imgVideo = cv2.resize(imgVideo, (wT, hT))
#
# orb = cv2.ORB_create(nfeatures=1000)
# kp1, des1 = orb.detectAndCompute(imgTarget, None)
# # imgTarget = cv2.drawKeypoints(imgTarget, kp1,None)
# img2 = np.zeros([wT, hT])
# # imgWarp = np.zeros([wT,hT])
# while True:
#
#     ret, imgWebcam = cap.read()
#     imgAug = imgWebcam.copy()
#     kp2, des2 = orb.detectAndCompute(imgWebcam, None)
#     if detection == False:
#         Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         frameCounter = 0
#     else:
#         if frameCounter == Video.get(cv2.CAP_PROP_FRAME_COUNT):
#             Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             frameCounter = 0
#         ret, imgVideo = Video.read()
#         imgVideo = cv2.resize(imgVideo, (wT, hT))
#
#     # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append(m)
#     print(len(good))
#     imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
#     if len(good) > 20:
#         detection = True
#         # ОБЪЯСНИТЬ
#         # https://docs.opencv.org/4.5.2/db/d27/tutorial_py_table_of_contents_feature2d.html
#         srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
#         print(matrix)
#
#         pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, matrix)
#         img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
#
#         imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
#
#         maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
#         cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
#         maskInv = cv2.bitwise_not(maskNew)
#         imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
#         imgAug = cv2.bitwise_or(imgWarp, imgAug)
#
#         # StackedImages = cvzone.stackImages([imgWebcam,imgWarp,imgTarget, imgFeatures,imgWarp,imgAug],3, 0.5)
#
#     cv2.imshow("features", imgAug)
#     cv2.imshow("img2", img2)
#     cv2.imshow("imgWarp", imgFeatures)
#     # cv2.imshow("maskNew", imgAug)
#     # cv2.imshow("target", imgTarget)
#     # cv2.imshow("webcam", imgWebcam)
#     # cv2.imshow("video", imgVideo)
#
#     cv2.waitKey(1)
#     frameCounter += 1

######### НЕ ОБЯЗАТЕЛЬНО
# filename = 'C:/users/alexm/Pictures/testAR3.jpg'
# img = cv.imread(filename)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
