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

```bash
git clone https://github.com/your-username/AD-Classification.git
cd AD-Classification
Install the required packages:

bash
Copy code
pip install torch torchvision pillow numpy matplotlib scikit-learn seaborn
Data Preparation
The script downloads and extracts the dataset from Kaggle.

Model Architecture
The CNN architecture consists of:

Convolutional layers for feature extraction
Max pooling layers for downsampling
ReLU activation functions
Fully connected (dense) layers for classification
Training and Validation
The dataset is split into training and validation sets (80-20 split). The model is trained using the CrossEntropyLoss and Adam optimizer. Early stopping is implemented to prevent overfitting.

Evaluation
The model's performance is evaluated using accuracy, precision, and recall metrics. A confusion matrix is also plotted to visualize the classification results.

Usage
To train the model, run the training script:

bash
Copy code
python train.py
To evaluate the model, run the evaluation script:

bash
Copy code
python evaluate.py
To predict on a new image, use the prediction script:

bash
Copy code
python predict.py --image_path path_to_image
Results
Training and validation loss, as well as validation accuracy, are plotted after training. The model achieves high accuracy in classifying the images into the respective categories.

