# Importing required libs
import glob
import cv2

import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers import Dropout, Dense
from keras.models import Sequential

from extract_bottleneck_features import *

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
 
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

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img, verbose=0))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
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

dog_names = ['train\\001.Affenpinscher',
 'train\\002.Afghan_hound',
 'train\\003.Airedale_terrier',
 'train\\004.Akita',
 'train\\005.Alaskan_malamute',
 'train\\006.American_eskimo_dog',
 'train\\007.American_foxhound',
 'train\\008.American_staffordshire_terrier',
 'train\\009.American_water_spaniel',
 'train\\010.Anatolian_shepherd_dog',
 'train\\011.Australian_cattle_dog',
 'train\\012.Australian_shepherd',
 'train\\013.Australian_terrier',
 'train\\014.Basenji',
 'train\\015.Basset_hound',
 'train\\016.Beagle',
 'train\\017.Bearded_collie',
 'train\\018.Beauceron',
 'train\\019.Bedlington_terrier',
 'train\\020.Belgian_malinois',
 'train\\021.Belgian_sheepdog',
 'train\\022.Belgian_tervuren',
 'train\\023.Bernese_mountain_dog',
 'train\\024.Bichon_frise',
 'train\\025.Black_and_tan_coonhound',
 'train\\026.Black_russian_terrier',
 'train\\027.Bloodhound',
 'train\\028.Bluetick_coonhound',
 'train\\029.Border_collie',
 'train\\030.Border_terrier',
 'train\\031.Borzoi',
 'train\\032.Boston_terrier',
 'train\\033.Bouvier_des_flandres',
 'train\\034.Boxer',
 'train\\035.Boykin_spaniel',
 'train\\036.Briard',
 'train\\037.Brittany',
 'train\\038.Brussels_griffon',
 'train\\039.Bull_terrier',
 'train\\040.Bulldog',
 'train\\041.Bullmastiff',
 'train\\042.Cairn_terrier',
 'train\\043.Canaan_dog',
 'train\\044.Cane_corso',
 'train\\045.Cardigan_welsh_corgi',
 'train\\046.Cavalier_king_charles_spaniel',
 'train\\047.Chesapeake_bay_retriever',
 'train\\048.Chihuahua',
 'train\\049.Chinese_crested',
 'train\\050.Chinese_shar-pei',
 'train\\051.Chow_chow',
 'train\\052.Clumber_spaniel',
 'train\\053.Cocker_spaniel',
 'train\\054.Collie',
 'train\\055.Curly-coated_retriever',
 'train\\056.Dachshund',
 'train\\057.Dalmatian',
 'train\\058.Dandie_dinmont_terrier',
 'train\\059.Doberman_pinscher',
 'train\\060.Dogue_de_bordeaux',
 'train\\061.English_cocker_spaniel',
 'train\\062.English_setter',
 'train\\063.English_springer_spaniel',
 'train\\064.English_toy_spaniel',
 'train\\065.Entlebucher_mountain_dog',
 'train\\066.Field_spaniel',
 'train\\067.Finnish_spitz',
 'train\\068.Flat-coated_retriever',
 'train\\069.French_bulldog',
 'train\\070.German_pinscher',
 'train\\071.German_shepherd_dog',
 'train\\072.German_shorthaired_pointer',
 'train\\073.German_wirehaired_pointer',
 'train\\074.Giant_schnauzer',
 'train\\075.Glen_of_imaal_terrier',
 'train\\076.Golden_retriever',
 'train\\077.Gordon_setter',
 'train\\078.Great_dane',
 'train\\079.Great_pyrenees',
 'train\\080.Greater_swiss_mountain_dog',
 'train\\081.Greyhound',
 'train\\082.Havanese',
 'train\\083.Ibizan_hound',
 'train\\084.Icelandic_sheepdog',
 'train\\085.Irish_red_and_white_setter',
 'train\\086.Irish_setter',
 'train\\087.Irish_terrier',
 'train\\088.Irish_water_spaniel',
 'train\\089.Irish_wolfhound',
 'train\\090.Italian_greyhound',
 'train\\091.Japanese_chin',
 'train\\092.Keeshond',
 'train\\093.Kerry_blue_terrier',
 'train\\094.Komondor',
 'train\\095.Kuvasz',
 'train\\096.Labrador_retriever',
 'train\\097.Lakeland_terrier',
 'train\\098.Leonberger',
 'train\\099.Lhasa_apso',
 'train\\100.Lowchen',
 'train\\101.Maltese',
 'train\\102.Manchester_terrier',
 'train\\103.Mastiff',
 'train\\104.Miniature_schnauzer',
 'train\\105.Neapolitan_mastiff',
 'train\\106.Newfoundland',
 'train\\107.Norfolk_terrier',
 'train\\108.Norwegian_buhund',
 'train\\109.Norwegian_elkhound',
 'train\\110.Norwegian_lundehund',
 'train\\111.Norwich_terrier',
 'train\\112.Nova_scotia_duck_tolling_retriever',
 'train\\113.Old_english_sheepdog',
 'train\\114.Otterhound',
 'train\\115.Papillon',
 'train\\116.Parson_russell_terrier',
 'train\\117.Pekingese',
 'train\\118.Pembroke_welsh_corgi',
 'train\\119.Petit_basset_griffon_vendeen',
 'train\\120.Pharaoh_hound',
 'train\\121.Plott',
 'train\\122.Pointer',
 'train\\123.Pomeranian',
 'train\\124.Poodle',
 'train\\125.Portuguese_water_dog',
 'train\\126.Saint_bernard',
 'train\\127.Silky_terrier',
 'train\\128.Smooth_fox_terrier',
 'train\\129.Tibetan_mastiff',
 'train\\130.Welsh_springer_spaniel',
 'train\\131.Wirehaired_pointing_griffon',
 'train\\132.Xoloitzcuintli',
 'train\\133.Yorkshire_terrier']

def ResNet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model_2.predict(bottleneck_feature, verbose=0)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def human_dog_identifier(img_path):
    
    # if dog_detector detects a dog
    if dog_detector(img_path) == True:
        print("hello, dog!")
        print("You look like a ...")
        print(ResNet50_predict_breed(img_path).split(".")[1])
    
    # if face_detector detects a human
    elif face_detector(img_path) == True:
        print("hello, human!")
        print("You look like a ...")
        print(ResNet50_predict_breed(img_path).split(".")[1])
    
    else:
        print("ERROR: Please ensure images are all face on, are you sure you are a human/dog?")