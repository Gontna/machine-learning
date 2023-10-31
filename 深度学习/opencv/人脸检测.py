import cv2
import numpy as np

face_cascada = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
img = cv2.imread('mans.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
facas = face_cascada.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in facas:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray= gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eyes_cascade.detectMultiScale(roi_gray)
    for ( ex,ey,ew, eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+ew),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()