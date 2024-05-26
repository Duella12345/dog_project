
# Dog Breed Classifer Project

![alt text](app_test_images/dog/dachshund-1519374_640.jpg)

### Summary
This project uses Convolutional Neural Networks (CNNs). In this project, I learnt how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

The project includes a Flask web app where a user can input a new image and get classification of human / dog / or an error if they are neither plus the closest resembling dog breed. 

## Files in Repository
 - Root Directory
    - app (files relating to the application and new predictions from model)
		- static
			- js
				-image_upload.js (javascript for uploading image)
        - templates (html templates)
            - index.html
			- result.html
        - app.py (flask app file)
		- dog_names.txt (list of dog names for model categories)
		- model.py (the app models)
    - app_test_images (folder containing images for final algorithm test)
	- haarcascades
	- images (images in notebook and readme)
    - notebook (notebooks used in developing the scripts)
        - dog_app.ipynb
        - extract_bottleneck_features.py
    - LICENSE.txt
	- requirements.txt (required libraries for running app and model)
	- train_model.py (python to train model prior to using in app)

Screenshot of notebook output:

![alt text](images/Screenshot_app.png)



## Project setup

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/Duella12345/dog_project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location 

`path/to/dog-project/dogImages`

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location 

`path/to/dog-project/lfw`

If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location 

`path/to/dog-project/bottleneck_features`

5. Download the [Resnet-50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset.  Place it in the repo, at location 

`path/to/dog-project/bottleneck_features`

6. Open the notebook and run all to see the development process.
```
jupyter notebook dog_app.ipynb
```

7. Run the following command in the project's root directory to set up model, this may take a few minutes to run (check requirements file for required libraries). To run the model training pipeline:
        
`python train_model.py`

2. To run the app: go to `app` directory: `cd app`

3. Run the web app: 

`flask --app app.py run`

4. Visit 'http://127.0.0.1:5000' to open the homepage

![alt text](images/Screenshot_homepage.png)

5. Click on 'choose file' and upload an image of a person or a dog and press submit (it may take a few moments)

![alt text](images/Screenshot_upload.png)

6. See your prediction!

![alt text](images/Screenshot_prediction.png)

## High-level Overview

In this project I have created an algorithm and used it to create a web app that accepts any user-supplied image as input.  If a dog is detected in the image, it provides an estimate of the dog's breed.  If a human is detected, it provides an estimate of the dog breed that they most resemble.  The image below displays potential sample output. 

![alt text](images/Screenshot_prediction.png)

The project pieces together a series of models to perform different tasks:

1. an algorithm that detects humans in an image 
2. an algorithm that detects a dog  in an image
3. an algorithm that infers dog breed

## Description of Input Data

The dataset used for the project, includes a set of dog images `../dogImages/train` and human images `../lfw/lfw/*/*`.

- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to dog images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels
- `human_files` - numpy arrays containing file paths to human images

## Strategy for solving the problem

This project approaches the problem by breaking the algorithm into 3 separate algorithms and then combining them into one final function.

### Discussion of the expected solution

My proposed solution is to use a CNN that uses transfer learning from a pretrained image recognition model to infer the final dog breed plus two other algorithms, one to detectif it is a dog image and one to detect if it is a human image these will be combined to create the final overall algorithm.

1. an algorithm that detects humans in an image - I use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). I have downloaded one of these detectors and stored it in the `haarcascades` directory

2. an algorithm that detects a dog in an image using a a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), this model will categorise the image and check if it is in the range of dog breed categories within the ResNet-50 model.
3. an algorithm that infers dog breed using a CNN that uses transfer learning from a pretrained image recognition model.

### Metrics with justification

I chose accuracy as the evaluation metric to assess the performance of this solution. Accuracy is how close the resulting categorisation is to the correct category. Mathematically, this can be stated as:

Accuracy= TP + TN / TP + TN + FP + FN

There are other possible metrics such as specificity and sensitivity, but there are no requirements within the context of this model for a higher proportion of true positives or true negatives.

### Data Preprocessing

The images were preprocessed by loading the RGB image as a PIL Image type, converting the PIL Image type to a 3D tensor with shape (224, 224, 3) then converting the 3D tensor to 4D tensor with shape (1, 224, 224, 3) and returning a 4D tensor. I did also do some data augmentation for one of the models (random flip and rotation) to expand the dataset, but this was not used in the final model.

### Modelling

The model I used in this project was a CNN, I started with a basic CNN which had low accuracy and then used a pre-trained VGG16 model that I retraiend using transfer learning, the final model was a RESNET-50 model that was trained further using transfer learning on the dataset training set and validated on the validation set.

## Hyperparameter Tuning

The models are trained using the sklearn fit function with categorical_crossentropy as the loss function and accuracy as the metric using several epochs, if the model improves its validation loss then the saved model is updated with the new model until all epochs are completed.


## Results and Comparison Table
SO there are three main models that we are looking at to do the final identification of dog breed, we can see here that CNN with no pretraining is the least effective, we then have a VGG16 pretrained model that has 70.57% accuracy and finally the Resnet Model with 80.98% accuracy. This is the best accuracy as it is the highest but not so high that it may be overfitted and is good for the context that the model will be used in.

| Model								 |Accuracy|
|------------------------------------|--------|
|No pretraining model Test accuracy: |12.92%|
|Pretrained VGG16 Model Test accuracy| 70.57% |
|Pretrained Resnet2 Model Test accuracy:| 80.98%|


## Improvements

Some of the limitations of this model is the slow speed it takes to classify an image, this may become even more of a problem when the model is running in the cloud, therefore it may be worth trying out a model using [MobileNets](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/) this would mean a much smaller model that could run faster and more efficiently even on a mobile device. I also think the app could be improved by making it more mobile friendly and making the UI more interesting, this could be done with custom CSS.

## Acknowledgment

Thank you to the Udacity Data Science Course team for providing me with the knowledge and skills to create this project, thank you to the StackOverflow, Flask and AI community for providing the documentation, support and knowledge to help me when I stuck.