# Importing required libs
import cv2

import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers import Dropout, Dense
from keras.models import Sequential


def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor), verbose=0)

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    
    return len(faces) > 0

#predict general image label using resnet
def ResNet50_predict_labels(img_path):

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img, verbose=0))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):

    prediction = ResNet50_predict_labels(img_path)
    
    #checks if category is a dog breed
    return ((prediction <= 268) & (prediction >= 151)) 

def path_to_tensor(img_path):

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)

    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)

#predict dog breed
def ResNet50_predict_breed(img_path):

    # extract dog names
    file = open("dog_names.txt", "r")
    dog_names = file.read()
    dog_names = dog_names.split(",")
    file.close()

    # Loading model
    # Create a new model instance
    ResNet50_model_2 = Sequential()
    #define the input shape
    ResNet50_model_2.add(Input((1, 1, 2048)))

    # convert features to a single 2048 element vector
    ResNet50_model_2.add(GlobalAveragePooling2D())

    #use dropout to reduce overfitting
    ResNet50_model_2.add(Dropout(0.2))

    # predict if image is one of the 133 categories
    ResNet50_model_2.add(Dense(133, activation='softmax'))

    # Load the previously saved weights
    ResNet50_model_2.load_weights('..\saved_models\weights.best.ResNet50.weights.h5')

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    # obtain predicted vector
    predicted_vector = ResNet50_model_2.predict(bottleneck_feature, verbose=0)

    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# overall dog identification function
def human_dog_identifier(img_path):
    
    # if dog_detector detects a dog
    if dog_detector(img_path) == True:
        return "Hello, dog! You look like a ... " + ResNet50_predict_breed(img_path).split(".")[1]
    
    # if face_detector detects a human
    elif face_detector(img_path) == True:
        return "Hello, human! You look like a ... " + ResNet50_predict_breed(img_path).split(".")[1]
    
    else:
        return "ERROR: Please ensure images are all face on, are you sure you are a human/dog?"