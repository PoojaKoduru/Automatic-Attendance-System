import cv2
import numpy as np
import face_recognition as fr
import os
import PIL
from datetime import datetime
# To enable access to folders

# capture the image
cap = cv2.VideoCapture(0)
# to get each frame while camera is on
while True:
    success, img = cap.read()
    print(str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "," + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cv2.imshow('Capture', img)
    cv2.waitKey(0)
    # resize image to decrease image size to speed the process
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    # convert to rgb
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = fr.face_locations(imgS)[0]
    # face_location returns the coordinates of top left,top right,bottom right,bottom left in order

    y1, x2, y2, x1 = facesCurFrame
    # to display rectangle over image, we are resizing the image back
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    # place rectangle over face location
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(x2-x1,y2-y1)
    cv2.imshow('face detection', img)
    cv2.waitKey(0)

