import cv2
import numpy as np
import face_recognition as fr
import os
import PIL
from datetime import datetime

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
    nam=os.path.splitext(cl)[0]
    classNames.append(nam)
    im = PIL.Image.open(f'{path}/{cl}')
    # to access only image name without extensions
    width,height = im.size
    print(width,height)
    #show image and its dimensions
    cv2.imshow(nam,curImg)
    cv2.waitKey(0)
