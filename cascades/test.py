import cv2
filename = '1.jpg'

def detect(filename):
    face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print (faces)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0) ,2)
    cv2.namedWindow('1')
    cv2.imshow('1', img)
    cv2.waitKey(0)

detect(filename)
