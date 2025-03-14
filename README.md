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


# Installation 
### Clone the repository
Clone the main branch of the repository
```bash  
git clone https://github.com/twitte01/288R_Capstone.git
```
### Install dependencies
Ensure you have Python 3.11.11 installed. 
To keep dependencies isolated, we recommend using a virtual environment. 
Once inside your virtual environment, install the required dependencies. 
```python 
pip install -r requirements.txt
```
Note if you are using a conda environment, it is recommended to use conda install as pip install can lead to installation issues and Jupyter kernel crashes. 
### Access Large Files
   #### Dataset + Preprocessed Data
   #### Models

# Running Instructions 
## Data Preprocessing
## EDA
## Baseline Models 
   ### Train
   Run the baseline_models.ipynb notebook. This will use the correct data pipeline to build the MFCC features dataset and train the models using grid search. It will plot graphs showing the results also.
## VGG 
   ### Training Initial Model
   To train the initial VGG-16 CNN model using our pre-trained architectures with initial hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_VGG/train.py
   ```
   This will save the model as best_model.pth and the training history as training_history.csv in models/CNN_VGG/checkpoints. 
   ### Testing Initial Model
   Once trained, the models can be evaluated on the test dataset by running 
   ```bash 
   python models/CNN_VGG/test.py
   ```
   The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png in models/CNN_VGG/results. The testing and training results are compared to the initial ResNet, EfficientNet, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   ### Hyperparameter Tuning with Random Search
   We performed hyperparameter tuning using random search. Note, since we are performing random search, you may end up with different random configurations. To run the tuning process:
   ```bash 
   python models/CNN_VGG/train_hyperparameter.py
   ```
   This script will generate multiple model configurations and save each model configuration in models/CNN_VGG/checkpoints. The tuned model with the best results will be saved as best_tuned_model.pth. 
   ### Analyzing Hyperparameter-Tuning Model Performance
   To evaluate and compare the performance of different CNN models, including initial models and those from hyperparameter tuning, use the 'Hyperparameter Tuning Results' section (VGG subsection) in the CNN_Results_Analysis.ipynb notebook in the notebooks folder. Running the train_hyperparameter.py script will also create a hyperparemeter_tuning_results.csv that contains the configuration of the highest performing model, train loss, train accuracy, validation loss, validation accuracy, cohen kappa score, and MCC score. 
   ### Testing the Best Model 
   Once the best model configuration is determined from the hyperparameter tuning, the model can be tested on the test set to evaluate performance by running
   ```bash 
   python models/CNN_VGG/test_hyperparameter.py
   ```
   Running test.py will create a confusion matrix named tuned_best_model_test_results.txt, tuned_best_model_confusion_matrix.png, and tuned_best_model_confusion_matrix.csv at this location: models/CNN_VGG/results
   ### Testing the Best Model on Noisy Data
   To test the best model (selected from the tuning results) on noisy data, run the 'Test Noisy Data: Models Trained on Clean Data' section in the CNN_Results_Analysis.ipynb notebook in the notebooks folder. 
   ### Training on Noisy Data
   After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_VGG/train_noise.py
   ```
   This will save the model as Noise_trained_VGG_model.pth and the training history as VGG_noise_training_history.csv in models/CNN_VGG/results that contains the epoch, training loss, validation loss, train accuracy, and validation accuracy. 
   ### Testing Model Trained on Noisy Data
   Once the noise-trained model is ready, test its performance by running: After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_VGG/test_noise.py
   ```
   The results in noise_trained_model_test_results.txt, noise_trained_model_confusion_matrix.png and noise_trained_model_confusion_matrix.csv in models/CNN_VGG/results. The testing and training results are compared to the VGG-16, ResNet-18, Bespoke and baseline models in the '???' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder.
## ResNet
   ### Training Initial Model
   To train the initial ResNet-18 CNN model using our pre-trained architectures and using random search to test hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_ResNet/train.py
   ```
   This will save the model as best_resnet.pth in models/CNN_ResNet/checkpoints. 
   ### Testing Initial Model
   Once trained, the models can be evaluated on the test dataset by running 
   ```bash 
   python models/CNN_ResNet/test.py
   ```
   This will test on the normal test set and the test set with added background noise. The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png, noise_test_results.txt, noise_confusion_matrix.csv and noise_confusion_matrix.png will be in models/CNN_ResNet/results. The testing and training results are compared to the initial VGG-16, EfficientNet, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   ### Hyperparameter Tuning 
   Hyperparameter Tuning is done in the train.py where it does a random search and saves the best model
   ### Test
   Running python models/CNN_ResNet/test.py will test on the normal test set and the test set with added background noise. The results in test_results.txt, confusion_matrix.csv and confusion_matrix.png, noise_test_results.txt, noise_confusion_matrix.csv and noise_confusion_matrix.png will be in models/CNN_ResNet/results. The testing and training results are compared to the initial VGG-16, EfficientNet, Bespoke and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
## EfficientNet 
   ### Training Initial Model
   To train the initial EfficientNet-B0 CNN model using our pre-trained architectures with initial hyperparameters inside the project root run: 
   ```bash 
   python models/CNN_EfficientNet/train.py
   ```
   This will save the model as best_model.pth and the training history as training_history.csv in models/CNN_ResNet/checkpoints. 
   ### Testing Initial Model
   Once trained, the models can be evaluated on the test dataset by running 
   ```bash 
   python models/CNN_EfficientNet/test.py
   ```
   The results are in test_results.txt, confusion_matrix.csv, and confusion_matrix.png in models/CNN_EfficientNet/results. The testing and training results are compared to the initial VGG-16, ResNet-18, Bespoke, and baseline models in the 'Initial Model Comparison' section of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder. 
   ### Hyperparameter Tuning with Random Search 
   We performed hyperparameter tuning using random search. Note, since we are performing random search, you may end up with different random configurations. To run the tuning process: 
   ```bash 
   python models/CNN_EfficientNet/train_hyperparameter.py
   ```
   This script will generate multiple model configurations and save each model configuration in models/CNN_EfficientNet/checkpoints. 
   ### Analyzing Hyperparameter-Tuning Model Performance 
   To evaluate and compare the performance of different CNN models, including initial models and those from hyperparameter tuning, use the 'Hyperparameter Tuning Results' section (EfficientNet subsection) in the CNN_Results_Analysis.ipynb notebook in the notebooks folder. 
   ### Testing the Best Model on Noisy Data 
   To test the best model (selected from the tuning results) on noisy data, run the 'Test Noisy Data: Models Trained on Clean Data' section in the CNN_Results_Analysis.ipynb notebook in the notebooks folder. 
   ### Training on Noisy Data 
   After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_EfficientNet/train_noisy.py
   ```
   This will save the model as best_model_noise.pth and the training history as training_history_noise.csv in models/CNN_ResNet/checkpoints. 
   ### Testing Model Trained on Noisy Data 
   Once the noise-trained model is ready, test its performance by running:
   After selecting the best hyperparameters, the final CNN model is trained using noisy data:
   ```bash 
   python models/CNN_EfficientNet/test_noisy.py
   ```
   The results in test_results_noise.txt, confusion_matrix_noise.csv and confusion_matrix_noise.png in models/CNN_EfficientNet/results. The testing and training results are compared to the VGG-16, ResNet-18, Bespoke and baseline models in the 'Test Noisy Data: Model Trained on Noisy Data' & 'Test Trimmed Data: Model Trained on Noisy Data' sections of the CNN_Exploratory_Analysis.ipynb notebook in the notebooks folder.
### Bespoke 
   #### Train
   #### Test
   #### Evaluate 
