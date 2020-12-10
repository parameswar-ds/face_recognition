import tensorflow as tf
model = tf.keras.models.load_model('/home/parameswar/Documents/face_rec/converted_keras/keras_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("/home/parameswar/Documents/face_rec/converted_keras/converted_model.tflite", "wb").write(tflite_model)