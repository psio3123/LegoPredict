# https://engmrk.com/kerasapplication-pre-trained-model/

import keras
import numpy as np
import json

#from keras.applications import inception_v3
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img

#inception_model = inception_v3.InceptionV3(weights="imagenet")

 # convert JSON key field string to DICT Integer
def jsonKeys2int(x):
      if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
      return x

def predict_inception_v3(input):
    start = time.time()
    im = Image.fromarray(input, 'RGB')

    # Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((299, 299))
    img_array = np.array(im)
    img_tensor = np.expand_dims(img_array,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)


    # preprocess for inception_v3
    processed_image_inception_v3 = inception_v3.preprocess_input(img_tensor)

    # inception_v3
    predictions_inception_v3 = inception_model.predict(processed_image_inception_v3)
    label_inception_v3 = decode_predictions(predictions_inception_v3)
    print("label_inception_v3 = ", label_inception_v3)
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))


def predict_mobilenet(input):
    # Convert the captured frame into RGB
    im = Image.fromarray(input, 'RGB')

    # Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224, 224))
    img_array = np.array(im)
    img_tensor = np.expand_dims(img_array,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #img_tensor /= 255.  # imshow expects values in the range [0, 1]

    start = time.time()
    predictions_my = mobilenet_model.predict(img_tensor)

    ende = time.time()
    y_classes = predictions_my.argmax(axis=-1)
    print('Mobilnet:', labels[y_classes[0]], predictions_my[0, y_classes[0]], '{:5.3f}s'.format(ende - start))


def predict_vgg16_CV2(input):
    # Convert the captured frame into RGB
    im = Image.fromarray(input, 'RGB')

    # Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224, 224))
    img_array = np.array(im)

    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_tensor = np.expand_dims(img_array, axis=0) # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)

    start = time.time()
    predictions_my = vgg16_model.predict(img_tensor)
    ende = time.time()
    y_classes = predictions_my.argmax(axis=-1)
    print('VGG16  :', y_classes[0], labels[y_classes[0]],predictions_my[0,y_classes[0]],'{:5.3f}s'.format(ende - start))

def extractFrames(  ):

    cap = cv2.VideoCapture(0)
    count = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        croppend_image = frame[100:324, 200:424].copy()
        cv2.rectangle(frame, (200, 100), (424, 324), (0, 255, 255), 2)
        cv2.imshow('Detection Aera', croppend_image)
        cv2.imshow('WebCam', frame)
        predict_vgg16_CV2(frame)
        #predict_mobilenet(croppend_image)
        #predict_inception_v3(croppend_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):   # s = save image and boxes to annotation file
            print('Read %d frame:')
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start = time.time()
    # load labels
    labels_file = "./models/labels_8classes.json"
    with open(labels_file) as f:
        labels = json.load(f, object_hook=jsonKeys2int)
    print(labels)

    print("reading model...")

    #mobilenet_model = load_model('./models/LegoTrainedMobileNet_epochs20.h5')
    vgg16_model = load_model('./models/LegoTrainedVGG16_classes8_epochs10.h5')

    ende = time.time()
    print('{:5.3f}s'.format(ende - start))
    extractFrames()







