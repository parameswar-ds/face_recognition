import numpy as np
import tensorflow as tf
import cv2,numpy,time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
TFLITE_MODEL="/home/parameswar/Documents/face_rec/face_recog__tflite_repo/converted_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
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
        face_roi=img[y:y+h, x:x+w]
        face_img = cv2.resize(face_roi, (224, 224))
        image_array = np.asarray(face_img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        interpreter.set_tensor(input_details[0]['index'], [normalized_image_array])
        # run the inference
        interpreter.invoke()
        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])
        n=(int(np.argmax(output_data)))
        if n==0:
            cv2.putText(img,"Alex",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(img,"Samy",(x,y),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),2)
        roi_color = img[y:y+h, x:x+w]
    end=time.time()
    f=1//(end-now)
    print("fps:",f)
    cv2.imshow('img',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()