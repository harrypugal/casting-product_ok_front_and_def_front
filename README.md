# Casting Defect Detection with Transfer Learning

## Project Overview
This project aims to identify casting defects in products using a convolutional neural network (CNN) model. The model leverages transfer learning 
from pre-trained models (e.g., ResNet152V2) to classify images as either **OK** or **Defective**.

## Dataset
- The dataset includes images in two categories:
  - `ok_front`: Images of non-defective products.
  - `def_front`: Images of defective products.
- Directory structure:

## Project Workflow
1. **Data Preprocessing**:
 - Images are loaded and augmented using `ImageDataGenerator` to improve model robustness.
2. **Model Selection**:
 - Multiple pre-trained models were considered, including ResNet152V2, Xception, and InceptionResNetV2.
 - Transfer learning layers were added to each base model for binary classification.
3. **Model Training and Evaluation**:
 - Models were evaluated based on metrics such as accuracy, AUC, precision, and recall.
 - Confusion matrix and ROC AUC score were used to assess performance.

## Results
- Best Model: `ResNet152V2`
- Test Accuracy: 98.5%
- ROC AUC: 0.99
- Confusion Matrix:
![Confusion Matrix](images/confusion_matrix.png)  <!-- Update with actual path if adding images -->

## Installation and Usage
1. Clone the repository:
 ```bash
 git clone <your_repo_url>
