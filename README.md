Overview
This project involves predicting rainfall using machine learning models. The models evaluated include Logistic Regression, XGBoost Classifier, and Support Vector Classifier (SVC). The project includes data cleaning, visualization, model training, and performance evaluation.

Data Preparation
Prior to model training, extensive data cleaning and visualization were performed, including:

Handling missing values
Feature scaling and encoding
Exploratory Data Analysis (EDA) to understand data distribution and relationships
Models
Three machine learning models were used for prediction:

Logistic Regression
XGBoost Classifier
Support Vector Classifier (SVC) with RBF kernel and probability estimation
Model Training and Evaluation
The models were trained on the dataset and evaluated using the following metrics:

from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Define models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

# Train and evaluate models
for i in range(3):
    models[i].fit(X, Y)

    print(f'{models[i]} : ')

    # Training predictions
    train_preds = models[i].predict_proba(X) 
    print('Training Accuracy : ', roc_auc_score(Y, train_preds[:,1]))

    # Validation predictions
    val_preds = models[i].predict_proba(X_val) 
    print('Validation Accuracy : ', roc_auc_score(Y_val, val_preds[:,1]))
    print()

# Plot confusion matrix for SVC model
ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()


Recall for Class 0: 0.67
Recall is lower for class 0, suggesting that out of all actual 0's, only 67% are being correctly classified as 0. The model is misclassifying more true negatives as false positives.

F1-Score for Class 0: 0.74
The F1-score balances precision and recall, useful when there's an imbalance between these metrics.

Class 1 Performance:
For class 1 (rainfall predicted), the model performs much better with high precision (0.85), recall (0.94), and F1-score (0.90), indicating better performance in predicting class 1.

Overall Accuracy: 85%
This indicates that 85% of the predictions made by the SVC on the validation set are correct.

Visualization
The project includes visualization of the confusion matrix for the SVC model to provide insights into classification performance.

Conclusion
The models demonstrated varying performance, with SVC achieving the highest overall accuracy. The analysis highlights strengths and areas for improvement in the model's predictions.

Requirements
Python 3.x
scikit-learn
XGBoost
matplotlib
To run the project, ensure you have the necessary libraries installed. You can use pip to install them:

bash
pip install scikit-learn xgboost matplotlib
