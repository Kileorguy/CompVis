import cv2 as cv
import pandas as pd
import pickle
from skimage.feature import hog
import numpy as np


haarcasade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)

file = open('./model.pkl', 'rb')
classifier = pickle.load(file)
file.close()

print(cv.CAP_PROP_FRAME_WIDTH)
print(cv.CAP_PROP_FRAME_HEIGHT)


frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

gamma = 0.2
while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    _, shadow_mask = cv.threshold(gray, 20, 0, cv.THRESH_BINARY)

    gray = cv.inpaint(gray, shadow_mask, inpaintRadius=7, flags=cv.INPAINT_TELEA)
    gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
    
    

    faces = haarcasade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    if not len(faces) < 1:
        for face_rect in faces:
            x, y, w, h = face_rect
            
            face_image = gray[y:y+w, x:x+h]
            resized = cv.resize(face_image,(128,128))

            hog_feature = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            test_data = pd.DataFrame(hog_feature).T

            result=classifier.predict(test_data)[0]

            print(result)
            if result == 0:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                result = 'No Mask'
            else:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                result = 'Mask'
    cv.imshow('Camera', frame)
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()