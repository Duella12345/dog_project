
# Dog Breed Classifer Project

![alt text](app_test_images/dog/dachshund-1519374_640.jpg)

### Summary
This project uses Convolutional Neural Networks (CNNs). In this project, I learnt how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

The project includes a Flask web app where a user can input a new image and get classification of human / dog / or an error if they are neither plus the closest resembling dog breed. 

## Files in Repository
 - Root Directory
    - app (files relating to the application and new predictions from model)
		- static
			- css
				- custom.css (styling file)
			- js
				-image_upload.js (javascript for uploading image)
        - templates
            - index.html
            - layout.html
			- result.html
		- uploads (folder for uploaded images)
        - app.py
		- dog_names.txt
		- model.py
    - app_test_images (folder containing images for final algorithm test)
	- haarcascades
	- images (images in notebook)
    - notebook (notebooks used in developing the scripts)
        - dog_app.ipynb
        - extract_bottleneck_features.py
    - LICENSE.txt
	- requirements.txt (required libraries for running app and model)
	- train_model.py

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

credit: created using https://geekpython.in/flask-app-for-image-recognition for app helper code and https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/ to help with uploading files

## High-level Overview

Provide a concise introduction that outlines the purpose and scope of the project. Clearly communicate the problem statement and the significance of addressing it.
## Description of Input Data

Explain the dataset used for the project, including its source, format, and any relevant details. Provide insights into the variables and their significance in relation to the problem being solved.

## Strategy for solving the problem

Describe the overall approach and methodology employed to tackle the problem. Discuss any specific techniques, algorithms, or models used and the rationale behind their selection.

## Discussion of the expected solution

Present a detailed explanation of the proposed solution, including the overall architecture or workflow. Discuss how the various components fit together to address the problem and achieve the desired outcome.

## Metrics with justification

Identify and explain the evaluation metrics used to assess the performance of the solution. Justify the choice of these metrics based on their relevance to the problem and their ability to measure success effectively.

## EDA

Conduct an exploratory data analysis and document the key findings. Discuss any patterns, trends, or insights discovered through visualizations, statistical summaries, or other analytical techniques.

## Data Preprocessing

Outline the steps taken to preprocess the data, including any cleaning, transformation, or feature engineering techniques employed. Provide a clear and concise description of each preprocessing step and its purpose.

## Modeling

Present the details of the chosen model or models used in the project. Explain the underlying algorithms and any specific considerations or modifications made. Include relevant code snippets or pseudocode for clarity.

## Hyperparameter Tuning

Describe the process of hyperparameter tuning for the selected model. Discuss the techniques used, such as grid search or random search, and explain the rationale behind the chosen hyperparameter values.

## Results

Present the results of the model evaluation and performance. Include relevant metrics, visualizations, or other outputs that demonstrate the effectiveness of the solution. Interpret the results and highlight key insights or observations.

## Comparison Table

Provide a comprehensive comparison table that compares the performance of different models or variations within the project. Include relevant metrics and other relevant information to facilitate easy comparison.
Conclusion: Summarize the key findings, insights, and conclusions derived from the project. Recap the main points and emphasize the success of the proposed solution. Discuss the implications or potential applications of the project's outcomes.

## Improvements

Identify any limitations, challenges, or areas for improvement in the project. Discuss potential enhancements or future directions that could further enhance the solution or address any remaining gaps.

## Acknowledgment

Acknowledge any individuals, organizations, or resources that have contributed to the project's success. Express gratitude and recognition for their support, guidance, or contributions.