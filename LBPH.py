import cv2
import os
import sys
import numpy as np


def face_rec():
    names = ['lnx','lwj']
 
    X,y =[],[]
    for i in os.walk("data/at/lnx/"):#遍历图片文件夹
        for j in i[2]:
            im = cv2.imread("data/at/lnx/"+j, cv2.IMREAD_GRAYSCALE)
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(0)
    for i in os.walk("data/at/lmj/"):#遍历图片文件夹
        for j in i[2]:
            im = cv2.imread("data/at/lmj/"+j, cv2.IMREAD_GRAYSCALE)
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(1)
    y = np.asarray(y, dtype=np.int32)



    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while(True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                if params[1] < 50:
                    print ("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                    cv2.putText(img, names[params[0]], (x, y -20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                else:
                    cv2.putText(img, 'others', (x, y -20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow("camera", img)
        if cv2.waitKey(256) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_rec()
    
    
    
