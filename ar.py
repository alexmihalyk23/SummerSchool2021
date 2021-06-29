import cv2
import numpy as np
import cvzone

imgTarget = cv2.imread("C:/users/alexm/Pictures/testAR3.jpg")
cap = cv2.VideoCapture(0)
Video = cv2.VideoCapture("C:/Users/alexm/Pictures/testAR2.mp4")

detection = False
frameCounter = 0

imgVideo = Video.read()[1]
hT,wT,cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT,hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1,None)
img2 = np.zeros([wT,hT])
# imgWarp = np.zeros([wT,hT])
while True:

    ret, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    if detection ==False:
        Video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    else:
        if frameCounter == Video.get(cv2.CAP_PROP_FRAME_COUNT):
            Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        ret, imgVideo = Video.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None, flags=2)

    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT], [wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        imgWarp = cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp,imgAug)
        # StackedImages = cvzone.stackImages([imgWebcam,imgWarp,imgTarget, imgFeatures,imgWarp,imgAug],3, 0.5)


    cv2.imshow("features", imgAug)
    cv2.imshow("img2", img2)
    cv2.imshow("imgWarp", imgFeatures)
    # cv2.imshow("maskNew", imgAug)
    # cv2.imshow("target", imgTarget)
    # cv2.imshow("webcam", imgWebcam)
    # cv2.imshow("video", imgVideo)

    cv2.waitKey(1)
    frameCounter+=1