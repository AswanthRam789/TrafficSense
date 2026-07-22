import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/trafficsense.keras")

img_path = input("Enter Image Path: ")

img = image.load_img(img_path, target_size=(64,64))

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] >= 0.5:
    print("\nPrediction : Vehicle")
    print("Confidence :", round(prediction[0][0]*100,2),"%")
else:
    print("\nPrediction : Non-Vehicle")
    print("Confidence :", round((1-prediction[0][0])*100,2),"%")