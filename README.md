# Classifying Audio Speech Commands
**DSC 288R Grad Capstone in Data Science**

**Group 14:** Parker Aman, Shane Lin, Ryan Thomson, Taylor Witte

## Table of Contents 
- [Description] (##Description )
- [Features] (##Features)
- [Installation] (##Installation)
- [Running Instruction] (#runninginstructions) 

## Description 


## Features 
### Data Preprocessing
### EDA
### Baseline Models
   #### Random Forest
   #### KNN
### CNN Models

   #### Bespoke
   #### Pretrained Models 
   We evaluated transfer learning with three pretrained CNN models, VGG-16, ResNet-18, and EfficientNet. 
   ##### VGG-16 
   
   ##### ResNet
   ##### EfficientNet


## Installation 
### Clone the repository
### Install dependencies
Ensure you have Python 3.11.11 installed. 
To keep dependencies isolated, we recommend using a virtual enviroment. 
Once inside your virtual enviroment, install the required dependencies. 
```python 
pip install -r requirements.txt
```
Note if you are using a conda enviroment, it is recommended to use conda install as pip install can lead to installation issues and jupyter kernel crashes. 
### Access Large Files
   #### Dataset + Preprocessed Data
   #### Models

## Running Instructions 
### Data Preprocessing
### EDA
### Random Forest Models 
   #### Train
   #### Test
   #### Evaluate 
### KNN 
   #### Train
   #### Test
   #### Evaluate 
### VGG 
   #### Training Initial Model
   To train the initial VGG-16 CNN model using our pre-trained architectures with initial hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_VGG/train.py
   ```
   This will save the model as best_model.pth and the training history as training_history.csv in models/CNN_VGG/checkpoints. 
   #### Testing Initial Model
   Once trained, the models can be evluated on the test dataset by running 
   ```bash 
   python models/CNN_VGG/test.py
   ```
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_VGG/results. The testing and training results are compared to the initial ResNet, EfficientNet, Bespoke and baseline models in the Initial Model Comparison Section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   #### Test
   #### Evaluate 
### ResNet
   #### Training Initial Model
   To train the initial ResNet-18 CNN model using our pre-trained architectures with initial hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_ResNet/train.py
   ```
   This will save the model as best_model.pth and the training history as training_history.csv in models/CNN_ResNet/checkpoints. 
   #### Testing Initial Model
   Once trained, the models can be evluated on the test dataset by running 
   ```bash 
   python models/CNN_ResNet/test.py
   ```
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_ResNet/results. The testing and training results are compared to the initial VGG-16, EfficientNet, Bespoke and baseline models in the Initial Model Comparison Section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   #### Hyperparameter Tuning 
   #### Test
   #### Evaluate 
### EfficientNet 
   #### Training Initial Model
   To train the initial EfficientNet-B0 CNN model using our pre-trained architectures with initial hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_EfficientNet/train.py
   ```
   This will save the model as best_model.pth and the training history as training_history.csv in models/CNN_ResNet/checkpoints. 
   #### Testing Initial Model
   Once trained, the models can be evluated on the test dataset by running 
   ```bash 
   python models/CNN_EfficientNet/test.py
   ```
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_EfficientNet/results. The testing and training results are compared to the initial VGG-16, ResNet-18, Bespoke and baseline models in the Initial Model Comparison Section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   #### Test
   #### Evaluate 
### Bespoke 
   #### Train
   #### Test
   #### Evaluate 
