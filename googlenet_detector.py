import cv2
import numpy as np
cap = cv2.VideoCapture(0)

model = "bvlc_googlenet.caffemodel"
config = "deploy.prototxt"
rows = open("labels_googleNet.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

googleNet = cv2.dnn.readNet(model,config)

while True:
    ret, frame  = cap.read()

    blob = cv2.dnn.blobFromImage(frame,1,(227,227))
    googleNet.setInput(blob)
    googlePreds = googleNet.forward()
    idx = np.argsort(googlePreds[0])[::-1][0]
    text = "Label: {}, {:.2f}%".format(classes[idx],googlePreds[0][idx] * 100)
    print(googlePreds[0].argmax())
    print(text)
    cv2.putText(frame, text,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


    cv2.imshow("net", frame)
    cv2.waitKey(1)