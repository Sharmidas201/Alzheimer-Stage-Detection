# Alzheimer's Disease Classification using CNN

This repository contains code for training and testing a Convolutional Neural Network (CNN) to classify Alzheimer's Disease (AD) from MRI images into four categories: MildDemented, ModerateDemented, NonDemented, and VeryMildDemented.

## Dataset

The dataset used in this project is the Alzheimer's Dataset (4 class of Images) available on Kaggle. The data is downloaded and unzipped directly in the code.

## Dependencies

- `torch`
- `torchvision`
- `PIL`
- `numpy`
- `matplotlib`
- `sklearn`
- `seaborn`
- `shutil`
- `tempfile`
- `urllib`

## Installation

First, clone this repository and navigate into the project directory:

Install the required packages:
pip install torch torchvision pillow numpy matplotlib scikit-learn seaborn

Data Preparation
The script downloads and extracts the dataset from Kaggle. The data preparation steps include:

Downloading and extracting the dataset.
Applying transformations like resizing, grayscaling, and normalization.
Splitting the dataset into training, validation, and test sets.
Model Architecture

The CNN architecture consists of:

Convolutional layers for feature extraction
Max pooling layers for downsampling
ReLU activation functions
Fully connected (dense) layers for classification
Training and Validation
The dataset is split into training,testing and validation sets . The model is trained using the CrossEntropyLoss and Adam optimizer. Early stopping is implemented to prevent overfitting. 

The training process includes:

Initializing the model
Defining the loss function and optimizer
Training the model for a specified number of epochs
Evaluating the model on the validation set
Saving the best model based on validation loss
Evaluation
The model's performance is evaluated using:

Accuracy, precision, and recall metrics for each class
Confusion matrix to visualize classification results
Plots of training and validation loss, and validation accuracy

The results include:

Training and validation loss plots
Validation accuracy plot
Confusion matrix for classification results
Metrics for each class (accuracy, precision, recall)


