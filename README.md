# Classifying Audio Speech Commands
**DSC 288R Grad Capstone in Data Science**

**Group 14:** Parker Aman, Shane Lin, Ryan Thomson, Taylor Witte

## Table of Contents 
- [Description] (##Description )
- [Features] (##Features)
- [Installation] (##Installation)
- [Running Instruction] (## Running Instructions) 

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
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_VGG/results. The testing and training results are compared to the initial ResNet, EfficientNet, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
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
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_ResNet/results. The testing and training results are compared to the initial VGG-16, EfficientNet, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
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
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_EfficientNet/results. The testing and training results are compared to the initial VGG-16, ResNet-18, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   #### Hyperparameter Tuning with Random Search 
   We performed hyperparameter tuning using random search. Note since we are performing random search, you may end up with different random configurations. To run the tuning process: 
   ```bash 
   python models/CNN_EfficientNet/train_hyperparameter.py
   ```
   This script will generate multiple model configurations and save each model configuration in models/CNN_EfficientNet/checkpoints. 
   #### Analyzing Hyperparameter-Tuning Model Performance 
   To evaluate and compare the performance of different CNN models, including initial models and those from hyperparameter tuning, use the 'Hyperparameter Tuning Results' section (EfficientNet subsection) in the CNN_Results_Anlaysis.ipynb notebook in the notebooks folder. 
   #### Testing the Best Model on Noisy Data 
   To test the best model (selected from the tuning results) on noisy data, run the 'Test Noisy Data: Models Trained on Clean Data' section in the CNN_Results_Analysis.ipynb notebook in the notebooks folder. 
   #### Training on Noisy Data 
   After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_EfficientNet/train_noisy.py
   ```
   This will save the model as best_model_noise.pth and the training history as training_history_noise.csv in models/CNN_ResNet/checkpoints. 
   #### Testing Model Trained on Noisy Data 
   Once the noise-trained model is ready, test its performance by running:
   After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_EfficientNet/test_noisy.py
   ```
   The results in test_results_noise.txt, confusion_matrix_noise.csv and confusion_matrix_noise.png in models/CNN_EfficientNet/results. The testing and training results are compared to the VGG-16, ResNet-18, Bespoke and baseline models in the '???' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder.
### Bespoke 
   #### Train
   #### Test
   #### Evaluate 
