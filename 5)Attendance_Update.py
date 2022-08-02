import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import *
import pandas as pd
#for text to speech
import pyttsx3
engine = pyttsx3.init()
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

# mark attendance
def markAttendance(name):
    # opening the attendance sheet
    with open('Attendance.csv', 'r+') as f:
        # a list to read each line in attendance one by one
        myDataList = f.readlines()
        nameList =[]
        print(myDataList)

        for line in myDataList:
            # each line in attendance file has name,time. so we are splitting to access first element name and storing in namelist
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            # if attendance not marked already, only then current time is accessed and attendance is updated
            now = datetime.now()
            dtString = now.strftime('%D %H:%M:%S')
            dt = now.strftime('%D')
            hr = dtString[9:11]
            #hour is checked to mark attendance for that particular class
            # if name is already not there for a particular day, it means that its their first class,so previous classes are left blank to indicate absent
            if int(hr)==int(9):
                f.writelines(f'\n{name},{dtString[0:8]},{dtString[9:17]}')
            if int(hr)==int(10):
                f.writelines(f'\n{name},{dtString[0:8]},,{dtString[9:17]}')
            if int(hr)==int(11):
                f.writelines(f'\n{name},{dtString[0:8]},,,{dtString[9:17]}')



        l = len(name)
        if name in nameList:
            i = 0
            for nam in nameList:
                if nam == name:
                    time = myDataList[i]
                    j=i

                i = i + 1
            #if name is already in attendance sheet , then last entry of date for that particular person is extracted

            now = datetime.now()
            presdt = now.strftime('%D ')
            prevdt = time
            m1, d1, y1 = [int(x) for x in presdt.split('/')]
            b1 = date(y1, m1, d1)
            m2, d2, y2 = [int(x) for x in time[l+1:l+9].split('-')]
            b2 = date(y2, m2, d2)
            if b1>b2 :
                #if current date is greater than last entry, it means new entry
                #so attendance is entered in a new line, with similar absentee indication as in case of name not in sheet
                now = datetime.now()
                dtString = now.strftime('%D %H:%M:%S')
                hr1 = dtString[9:11]
                if(int(hr1)==9):
                    f.writelines(f'\n{name},{dtString[0:8]},{dtString[9:17]}')
                if (int(hr1) == 10):
                    f.writelines(f'\n{name},{dtString[0:8]},,{dtString[9:17]}')
                if (int(hr1) == 11):
                    f.writelines(f'\n{name},{dtString[0:8]},,,{dtString[9:17]}')

            if b1==b2 :
                #if the date is same then hour is extracted
                #Based on hour only a particular coloumn is updated ,no new row is created
                dtStr = now.strftime('%D %H:%M:%S')
                hr1=dtStr[9:11]
                if(int(hr1)==10 ):
                    df = pd.read_csv("Attendance.csv")
                    df.loc[j-1, 'Lecture2'] = dtStr[9:17]
                    df.to_csv("Attendance.csv", index=False)
                if (int(hr1) == 11):
                    df = pd.read_csv("Attendance.csv")
                    df.loc[j - 1, 'Lecture3'] = dtStr[9:17]
                    df.to_csv("Attendance.csv", index=False)




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
        j=0
        for i in classNames:
            print(i," : ", faceDis[j])
            j=j+1
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
            #spell out the name
            engine.say(name)
            engine.say("Your attendance has been marked successfully, Please get into the class")
            engine.runAndWait()
            #and mark the attendance
            markAttendance(name)
            # display image
            cv2.imshow('Webcam', img)
            cv2.waitKey(0)
        else:
            #if its a new person, do registration
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
            #warning message to say it as an unauthorised access
            engine.say("Unknown person detected")
            engine.runAndWait()
            print("WARNING : unknown face detected")
            #password protection to ensure legit registrations only
            o= int(input("password"))
            if(o==1234):
                #if password is correct save the picture
                path = "database"
                cap = cv2.VideoCapture(0)
                # to get each frame while camera is on
                i = 0
                success, img = cap.read()
                while i == 0:
                    f = input("enter name")
                    fp = f + ".jpg"
                    os.chdir(path)
                    cv2.imwrite(fp, img)
                    break
            else:
                #if wrong password entered, then alert everyone
                engine.say("Unauthorised request")
                engine.runAndWait()
                print("unauthorised registration")









