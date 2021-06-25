import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haar.xml")

# rows, cols, channels = mask_img.shape



# Now black-out the area of logo in ROI

while True:

    ret, frame = cap.read()

    # CODE HERE
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)

    # в нашей переменной теперь хранятся хначения координат лица


    # threshold это пороговое значение
    # это значит что все пиксели, которые меньше порога становятся 0 а все что больше - 255
    # для того чтобы это работало корректно, сначала нам нужно перевести изображение в оттенки серого
    mask_img = cv2.imread("mask.png")
    img2gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
    for (x, y, w, h) in faces:
        # print(rows,cols,w,h)
        roi = frame[y+40:y+40+h, x:x+w]
        # mask_img = cv2.resize(mask_img,(w,h))
        # mask1 = mask.copy()

        mask = cv2.resize(mask,(w,h))
        mask_img = cv2.resize(mask_img,(w,h))
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(mask_img, mask_img, mask=mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)


        dst = cv2.resize(dst, (w,h))
        frame[y+40:y+40+h, x:x+w] = dst
    # for (x, y, w, h) in faces:
    #     mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     ret, mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY)
    #     mask_inv = cv2.bitwise_not(mask)
    #     ROI = frame[y:y+h,x:x+w]
    #     img1_bg = cv2.bitwise_and(ROI, ROI, mask=mask_inv)
    #     # Take only region of logo from logo image.
    #     img2_fg = cv2.bitwise_and(frame, frame, mask=mask)
    #     # Put logo in ROI and modify the main image
    #     dst = cv2.add(img1_bg, img2_fg)
    #     frame[y:y+h, x:x+h] = dst


    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("img", frame)
    keyCode = cv2.waitKey(1)

    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
