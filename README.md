# Heart Disease Classification
The project was carried out to complete the final exam of the machine learning course.

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Contribution](#contribution)

## Description
This project uses the `heart.csv` dataset to build a predictive model that can determine the likelihood of a person having heart disease based on various features such as age, gender, blood pressure, cholesterol, and more.

## Requirements
This project is fully developed and executed in Google Colab. Make sure you have a Google account to access Colab.

## Usage
1. Open Google Colab and create a new notebook or open an existing notebook.
2. Upload the `heart.csv` file to Colab:
    - Click the folder icon on the left to open the file tab, then upload the `heart-disease-classification file`.
    - Click the upload button and select `heart.csv` from your local directory.
    - Run the model program.
      
## Results
### Cross Validation Score
- **Cross Validation Scores:** 0.9372815748257143
  - This is the average accuracy obtained from cross-validation, indicating that the model has consistent performance with an accuracy of about 93.73%.

### Best Parameters
- **Best Parameters:** SVC(C=50, gamma=0.1, probability=True, random_state=128)
  - The best parameters found by GridSearchCV for the SVM model, including the values for C, gamma, and other settings.

### Train and Test Accuracy
- **Train Accuracy of Classifier:** 1.0
  - The model has 100% accuracy on the training data, indicating that the model is very good at recognizing patterns in the training data.
- **Test Accuracy of Classifier:** 0.9666666666666667
  - The accuracy on the test data is about 96.67%, indicating that the model also performs well on unseen data.

### Classification Report
The Classification Report provides evaluation metrics such as precision, recall, and f1-score for each class (0 = No Disease, 1 = Disease). Precision, recall, and f1-score for both classes are high, indicating that the model is balanced in classifying both classes.

### ROC Curve
The ROC Curve shows the model's performance in terms of the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR). Area Under the Curve (AUC) = 0.99, a very high AUC value indicates that the model has excellent discriminatory ability between positive and negative classes.

### Confusion Matrix
The Confusion Matrix provides a visual representation of the model's predictions versus the actual values. The matrix shows the number of correct and incorrect predictions for each class. Here, the model correctly classifies most samples with only a few errors.

### Conclusion
The results of this project show that the SVM model used has excellent performance in detecting heart disease. With an average cross-validation score of 93.73%, the model demonstrates strong consistency in accuracy. The best parameters found by GridSearchCV ensure that the model is well optimized. The perfect training accuracy (100%) and high test accuracy (96.67%) indicate that this model is highly effective in recognizing patterns both in the training data and in previously unseen data. The classification report indicates that the model has high precision, recall, and f1-score values for both classes, showing a good balance in classification. The ROC curve with an AUC of 0.99 highlights the model's excellent discriminatory ability between positive and negative classes. The confusion matrix confirms that the model makes few prediction errors, reinforcing the validity and reliability of the classification results.

## Directory Structure
- `heart.csv`: The dataset used for training and evaluating the model
- `heart_disease_classification.ipynb`: Source code for the classification model
- `README.md`: This document

## Contribution
Contributions are welcome! To get started, follow these steps:
1. Fork this repository
2. Create a new feature branch (`git checkout -b new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin new-feature`)
5. Create a Pull Request

