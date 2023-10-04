# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys

# define a video capture object
model = tf.keras.models.load_model('keras_model.h5')
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    check, frame = vid.read()
    img = cv2.resize(frame, (224,224))

    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
      
    normalised_image = test_image / 225.0

    predictions = model.predict(normalised_image)

    predicted_class = np.argmax(predictions, axis=1)

    class_labels = {0: "Classe 0", 1: "Classe1"}
    print(class_labels)

    label = class_labels.get(predicted_class[0], "Classe desconhecida")

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, label, (10, 50), font, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Resultado", frame)


    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()