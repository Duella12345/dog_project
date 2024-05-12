from sklearn.datasets import load_files       
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers import Dropout, Dense
from keras.models import Sequential
import numpy as np

# Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogResNet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
_, train_targets = load_dataset('dogImages/train')
_, valid_targets = load_dataset('dogImages/valid')


# Define your architecture.
ResNet50_model_2 = Sequential()

#define the input shape
ResNet50_model_2.add(Input(train_ResNet50.shape[1:]))

# convert features to a single 2048 element vector
ResNet50_model_2.add(GlobalAveragePooling2D())

#use dropout to reduce overfitting
ResNet50_model_2.add(Dropout(0.2))

# predict if image is one of the 133 categories
ResNet50_model_2.add(Dense(133, activation='softmax'))

# Compile the model.
ResNet50_model_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.weights.h5', save_weights_only=True,
                               verbose=1, save_best_only=True)

ResNet50_model_2.fit(train_ResNet50, train_targets, 
          validation_data=(valid_ResNet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)