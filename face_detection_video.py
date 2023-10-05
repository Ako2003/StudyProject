#Face Detection for real time and for video
import cv2

trained_face_data = cv2.CascadeClassifier('haarcade_frontalface_default.xml')#file that allows to identify people's faces

webcam = cv2.VideoCapture(0)#If variable is 0 it turns on your camera if it has a path to video it detect faces in video

while True:
    succesful_frame_read, frame =  webcam.read()#allows us to use webcam

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#makes video black-white style

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)#gives coordinates of faces

    for i in range(0, len(face_coordinates)):#identify amount of faces and draw rectangle around them
        (x,y,w,h) = face_coordinates[i ]
        cv2.rectangle(frame,(x, y),(x + w,y + h),(0,255,0),2)
    
    cv2.imshow('Clever Programmer Face Detector', frame)#create a window eith name 'Clever Programmer Face Detector'

    key = cv2.waitKey(10)#speed of a video

    if key == 81 or key == 113 or key == 32:#if you press 'q' or 'Q' or 'Space' it will stop the code
        break


webcam.release()#stops webcam

