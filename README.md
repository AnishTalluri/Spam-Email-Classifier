# Spam Email Classifier

A comparative study of linear classification algorithms for email spam detection using the Spambase dataset.

## Overview

This project implements and compares three linear classification models to distinguish spam from legitimate emails. The models are trained on 4,601 email samples with 57 extracted features including word frequencies, character frequencies, and capital letter statistics.

## Features

- **Perceptron Implementation**: Custom-built perceptron classifier from scratch
- **SVM Classification**: Linear Support Vector Machine with regularization tuning
- **Logistic Regression**: Probabilistic classifier with L2 regularization
- **Hyperparameter Optimization**: Systematic tuning using development set validation
- **Performance Analysis**: Confusion matrices and detailed error analysis

## Dataset

- **Source**: UCI Machine Learning Repository - Spambase Dataset
- **Samples**: 4,601 emails
- **Features**: 57 numeric attributes
- **Classes**: Binary (spam/not spam)
- **Split**: 80% training, 10% development, 10% test

## Models & Results

| Model | Best Hyperparameters | Test Accuracy |
|-------|---------------------|---------------|
| Perceptron | max_iterations=500 | 90.02% |
| Linear SVC | C=1.0 | 91.54% |
| Logistic Regression | C=10.0, penalty=l2 | 91.32% |

## Key Insights

- **Linear SVC** achieved the best performance with 91.54% test accuracy
- All models showed strong generalization with minimal overfitting
- Feature engineering from email metadata proved highly effective for classification

## Technologies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

```python
# Train the best performing model
from sklearn.svm import LinearSVC

model = LinearSVC(C=1.0, dual=False, max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
