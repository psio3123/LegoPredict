# https://engmrk.com/kerasapplication-pre-trained-model/

import keras
import numpy as np

from keras.applications import inception_v3
from keras.models 		import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator, load_img

inception_model = inception_v3.InceptionV3(weights="imagenet")


#from pygments.lexer import include

#os.environ["HTTPS_PROXY"] = "http://proxy.le.grp:8080"

def read_model():
	model = load_model('./LegoTrainedFineTuning.h5')
    #model = load_model('./trained_models/LegoTrainedFineTuning.h5')
	print(model.summary())
	return model


def show_results():
    # Show the results
    for i in range(len(predicted_classes)):
        pred_class = np.argmax(predictions[predicted_classes[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[predicted_classes[i]].split('/')[0],
            pred_label,
            predictions[predicted_classes[i]][pred_class])

        original = load_img('{}/{}'.format(validation_dir, fnames[predicted_classes[i]]))
        plt.figure(figsize=[7, 7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()



def get_pil_image(filename):
    # load an image in PIL format
    original_image = load_img(filename, target_size=(224, 224))

    # convert the PIL image (width, height) to a NumPy array (height, width, channel)
    numpy_image = img_to_array(original_image)

    # Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
    input_image = np.expand_dims(numpy_image, axis=0)

    print("PIL image size = ", original_image.size)
    print("NumPy image size = ", numpy_image.shape)
    print("Input image size = ", input_image.shape)
    plt.imshow(np.uint8(input_image[0]))
    return input_image



def predict_inception_v3( input_image ):
    # Load the Inception_V3 model


    # preprocess for inception_v3
    processed_image_inception_v3 = inception_v3.preprocess_input(input_image.copy())

    # inception_v3
    start = time.time()
    predictions_inception_v3 = inception_model.predict(processed_image_inception_v3)
    label_inception_v3 = decode_predictions(predictions_inception_v3)
    print("label_inception_v3 = ", label_inception_v3)
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))

def predict_mymodel( input_image ):

    start = time.time()
    predictions_my = my_model.predict(input_image.copy())
    print(predictions_my)
    #label_MY = decode_predictions(predictions_my)
    #print("My labels = ", label_MY )
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))
    y_classes = predictions_my.argmax(axis=-1)
    print('Best Match', y_classes)  # Newly trained model prediction. [0,1] = [cat, dog].



if __name__ == '__main__':

    my_model = read_model()

    filename = "./lego_fotos/validation/1x15_Blue/DSC_1221.jpg"
    lego_brick = get_pil_image(filename)

    filename = "./lego_fotos/validation/connector_black/DSC_1182.JPG"
    image_rose = get_pil_image(filename)

    #---------------------------------------------------------------------------------------------
    # Predict SingleImage via Inception V3
    #---------------------------------------------------------------------------------------------

    print("Inception - Lego Brick ")
    predict_inception_v3( lego_brick )
    print("Inception - Lego Connector Black ")
    predict_inception_v3( image_rose )


    #---------------------------------------------------------------------------------------------
    # My Trained Model - Single Image
    #---------------------------------------------------------------------------------------------
    # processed_image_my = my_model.preprocess_input(input_image.copy())

    print("My Model - Lego Brick ")
    predict_mymodel(lego_brick)

    print("Inception - Lego Connector Black ")
    predict_mymodel(image_rose)


    # ---------------------------------------------------------------------------------------------
    # My Trained Model - Batch
    # ---------------------------------------------------------------------------------------------
    print("My Model - BATCH ")
    start = time.time()
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_dir = './lego_fotos/predict/'
    image_size = 224
    val_batchsize = 10

    # Create a generator for prediction
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    print("Lables",idx2label )

    # Get the predictions from the model using the generator
    predictions = my_model.predict_generator(validation_generator,
                                          steps=validation_generator.samples / validation_generator.batch_size,
                                          verbose=1)

    predicted_classes = np.argmax(predictions, axis=1)

    #print("Predicted Classes",predicted_classes)


    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

    errors = predictions
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])

        original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
        plt.figure(figsize=[7, 7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))




