# Credit-Card-Fraud-Detection
This repository contains a machine learning-based application that detects fraudulent credit card transactions. The project leverages machine learning algorithms like Logistic Regression and Decision Tree Classifiers, and handles imbalanced data using SMOTE. The model is trained using transaction data and can classify transactions as either Normal or Fraudulent.

**Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data 

# Features
- **Data Preprocessing:** Standardizes the Amount feature and removes duplicates from the dataset.
  
- **Handling Imbalanced Data:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
  
- **Model Training:** Trains models like Logistic Regression and Decision Tree Classifier.
  
- **Model Evaluation:** Calculates Accuracy, F1 Score, Precision, and Recall to evaluate the model's performance.
  
- **Save/Load Model:** Trained models are saved as .pkl files using Joblib for future use.
 
- **Transaction Prediction:** Predicts whether a new transaction is fraudulent or not based on user input.

 # Repository Structure
![image](https://github.com/user-attachments/assets/3604ba7b-cf28-475a-9ba2-0cbf7f983829)

# Model Training
This project uses two models for classification:

- Logistic Regression
- Decision Tree Classifier
  
The models are evaluated using accuracy, F1 score, precision, and recall. Once trained, the model is saved using joblib and can be reused for future predictions.

# Output:

![image](https://github.com/user-attachments/assets/cbaf6a56-7168-4186-ad7a-23b27c130889)

## Using Logistic Regression Classifier

![image](https://github.com/user-attachments/assets/324f6338-ddd5-49a2-b6af-b54eed61c1d7)

## Using Decision Tree Classifier

![image](https://github.com/user-attachments/assets/363a108a-8a59-4ee5-8cec-76adbe8d28de)

![image](https://github.com/user-attachments/assets/5d068028-c43d-4058-83be-e2e38367cf1c)

