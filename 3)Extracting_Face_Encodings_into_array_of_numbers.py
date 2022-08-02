import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

# capture the image
cap = cv2.VideoCapture(0)
# to get each frame while camera is on
i = 0
while True:
    success, img = cap.read()
    # resize image to decrease image size to speed the process
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    # convert to rgb
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # to find multiple faces in image captured
    facesCurFrame = fr.face_locations(imgS)    # face_location returns the coordinates of top left,top right,bottom right,bottom left in order

    # to find encodings of each face in frame
    encodeCurFrame = fr.face_encodings(img)
    while i == 0:
        print(encodeCurFrame)
        i = 1
    y1, x2, y2, x1 = facesCurFrame
    # to display rectangle over image, we are resizing the image back
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    # place rectangle over face location
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('face detection', img)
    cv2.waitKey(0)
