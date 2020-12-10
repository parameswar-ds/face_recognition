import numpy as np
# import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras
import cv2,numpy,time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
m = tensorflow.keras.models.load_model('/home/parameswar/Documents/face_rec/converted_keras/keras_model.h5')
faces_list=["Alex","Samy"]
cap = cv2.VideoCapture("videoplayback (2) (online-video-cutter.com).mp4")
while cap.isOpened():
    ret, frame = cap.read()
    now=time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #
        size = (224, 224)
        face_roi=img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, (224, 224))
        face_img = face_img.reshape(1, 224, 224, 3)
        image_array = np.asarray(face_img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # data[0] = normalized_image_array
        prediction = m.predict(normalized_image_array)
        n=int(np.argmax(prediction))
        if n==0:
            cv2.putText(img,"Alex",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(img,"Samy",(x,y),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),2)

        #
        # roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # cc=cc+1
        # if c%3==0:
        #     cv2.imwrite(f"/home/parameswar/Documents/face_rec/rough/{str(c)}_{str(cc)}.jpg",roi_color)
    end=time.time()
    f=1//(end-now)
    print("fps:",f)
    cv2.imshow('img',frame)
        
    c=c+1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()