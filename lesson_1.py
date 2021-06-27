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

img1 = cv2.imread('C:/Users/alexm/Pictures/english_1.png')
img2 = cv2.imread('test.png')
img2 = cv2.resize(img2, (160,120))
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# threshold это пороговое значение
# это значит что все пиксели, которые меньше порога становятся 0 а все что больше - 255
# для того чтобы это работало корректно, сначала нам нужно перевести изображение в оттенки серого
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
print(mask)
mask_inv = cv2.bitwise_not(mask)
print(mask_inv)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()