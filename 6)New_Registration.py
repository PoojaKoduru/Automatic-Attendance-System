import cv2
import os
path= "database"
cap = cv2.VideoCapture(0)
# to get each frame while camera is on
i = 0
success, img = cap.read()
while i==0:
    #to get name
    f = input("enter name")
    fp = f+".jpg"
    #change path to database
    os.chdir(path)
    #save the new registration
    cv2.imwrite(fp, img)
    break
