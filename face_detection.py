#This program detect face in images
import cv2

trained_face_data = cv2.CascadeClassifier('haarcade_frontalface_default.xml')#this file allows to identify people's faces

img = cv2.imread('robert-downey-jr-medium.png')#path to image

grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#makes photo white-black style

face_coordinates = trained_face_data.detectMultiScale(grayscale_img)#gives people's faces coordinates

for i in range(0, len(face_coordinates)):#identify amount of faces and draw rectangle around them
    (x,y,w,h) = face_coordinates[i]
    cv2.rectangle(img,(x, y),(x + w,y + h),(0,255,0),2)

screen = cv2.imshow('Clever Programmer Face Detecter', grayscale_img)#creates window with name 'Clever Programmer Face Detecter'
cv2.waitKey()#waits until you press any button on keyboard
