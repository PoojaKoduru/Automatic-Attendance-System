import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
# To enable access to folders


# loading image from database
# Path of database of folders
path = 'database'
# access all images path from database
myList = os.listdir(path)
# to store images in list
images = []
# to store the respective image's name/ID
classNames = []
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # to access only image name without extensions


# Feature extraction
def findEncodings(images):
    encodeList= []
    # to store encodings in a list
    for img in images:
        # convert from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # finding encodings of images
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)



# capture the image
cap = cv2.VideoCapture(0)
# to get each frame while camera is on
while True:
    success, img = cap.read()
    # resize image to decrease image size to speed the process
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    # convert to rgb
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # to find multiple faces in image captured
    facesCurFrame = fr.face_locations(imgS)
    # face_location returns the coordinates of top left,top right,bottom right,bottom left in order

    # to find encodings of each face in frame
    encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        # compare captured image against each image in database and store matches
        matches = fr.compare_faces(encodeListKnown, encodeFace)

        # find distance between both encodings
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # the correct image will have lowest distance
        matchIndex = np.argmin(faceDis)

        # when match is found face is recognised
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            # to display rectangle over image, we are resizing the image back
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # place rectangle over face location
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # place rectangle below face to put text
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            # display image
            cv2.imshow('Webcam', img)
            cv2.waitKey(0)
        else:
            y1, x2, y2, x1 = faceLoc
            # to display rectangle over image, we are resizing the image back
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # place rectangle over face location
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # place rectangle below face to put text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            # display image
            cv2.imshow('Webcam', img)
            cv2.waitKey(0)
            print("WARNING : unknown face detected")


