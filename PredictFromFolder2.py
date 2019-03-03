import os
import tkinter as tk
from tkinter import filedialog
import keras
import numpy as np
import json

from keras.applications import inception_v3
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img

import matplotlib.pyplot as plt
import os
import time


root = tk.Tk()
root.withdraw()


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

def get_filenames(root):
    root.withdraw()
    print("Initializing Dialogue... \nPlease select a file.")
    tk_filenames = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes = [('Images', '.jpg'), ('all files', '*.*'),], title='Please select one or more files')
    filenames = list(tk_filenames)
    return filenames

def predict_mymodel(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    start = time.time()
    predictions_my = model.predict(img_tensor)
    ende = time.time()
    y_classes = predictions_my.argmax(axis=-1)
    print('Best Match', labels[y_classes[0]], predictions_my[0, y_classes[0]], '{:5.3f}s'.format(ende - start))


if __name__ == '__main__':
    labels_file = "./models/labels_classes6.json"
    with open(labels_file) as f:
        labels = json.load(f, object_hook=jsonKeys2int)
    print(labels)

    print("reading model...")
    model = load_model('./models/LegoTrainedVGG16_15Layer_classes6_best_model.h5')

    while (True):
        images = get_filenames(root)
    
        for image_path in images:
            print("Image", image)
            predict_mymodel(image_path, model)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break



