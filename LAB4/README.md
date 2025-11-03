# Experiment Tracking: MLFlow

## Source Lab:
[https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1)

## MLFlow.ipynb
1. Notebook that runs 2 models - Logistic Regression and a simple Neural Network and iterates over hyperparmeters.
2. Dataset: Synthetic Binary classification dataset from sklearn with 2 input features
3. After model tracking, we get the model with highest recall and tests the model by (1) loading it and (2) serving it

## Key concepts
1. mlflow.log_params() - to log hyperparameters
2. mlflow.log_model() - to save model artifacts
3. mlflow.lod_metrics() - metrics like accuracy, f1 scores, etc.

## Setup
```
pip install mlflow scikit-learn torch numpy
```

*Note: I faced issues when importing mlflow with python 3.9. I upgraded python to 3.13*