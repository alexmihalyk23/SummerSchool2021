import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haar.xml")
mask = cv2.imread("test.png", -1)
# mask1 = mask.copy()
# mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
def transparentOverlay(src, overlay , pos = (0,0)  , scale = 1):
    overlay = cv2.resize(overlay , (0,0) ,fx = scale , fy = scale)
    h , w , _ =  overlay.shape ## size of foreground image
    rows , cols , _ = src.shape  ## size of background image
    y , x = pos[0] , pos [1]


    for i in range(h):
        for j in range(w):
            if x + i > rows or y + j >=cols:
                continue
            alpha = float(overlay[i][j][3]/255) ##  read the alpha chanel
            src[x+i][y+j] = alpha * overlay[i][j][:3] + (1-alpha) * src[x+i][y+j]
    return src



while True:

    ret, frame = cap.read()

    # CODE HERE
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=12)
    # в нашей переменной теперь хранятся хначения координат лица
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:

            mask_symin = int(y+0.1 * h / 6)
            mask_symax = int(y + 8* h / 6)
            sh_mask = mask_symax - mask_symin

            face_glass_ori = frame[mask_symin:mask_symax, x:x + w]

            mask = cv2.resize(mask, (w, sh_mask))
            transparentOverlay(face_glass_ori, mask)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("img", frame)
    keyCode = cv2.waitKey(1)

    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
