import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cvzone
cap = cv2.VideoCapture(0)

mask = cv2.imread("test.png")
segmentator = SelfiSegmentation()
mask = cv2.resize(mask, (640,480))

while True:
    ret, frame = cap.read()
    imgOut = segmentator.removeBG(frame, mask, threshold=0.8)

    imgStack = cvzone.stackImages([frame, imgOut],2,1)

    cv2.imshow("image", imgStack)
    cv2.waitKey(1)