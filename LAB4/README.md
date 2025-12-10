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

## Revision 2: Codebase Structure

The notebook has been refactored into a modular Python codebase for better organization and reusability. The codebase follows a simple, prototype-style structure focusing on core MLflow concepts.

### Project Structure
```
LAB4/
├── src/
│   ├── data.py          # Synthetic dataset generation
│   ├── models.py        # Model definitions (LR and NN)
│   ├── train.py         # Training with hyperparameter search
│   ├── evaluate.py      # Find best model by recall
│   ├── predict.py       # Load model and make predictions
│   └── serve.py         # Save model for serving
├── MLFlow.ipynb         # Original notebook
├── test_input.json      # Sample input for model serving
└── requirements.txt     # Python dependencies
```

### Codebase Components

**`src/data.py`**
- `generate_data()`: Creates synthetic binary classification dataset with 2 features
- Returns feature matrix X and target vector y

**`src/models.py`**
- `create_lr_model()`: Factory function for Logistic Regression with configurable hyperparameters
- `SimpleNN`: PyTorch neural network class (2 → 16 → 1 architecture)
- `create_nn_model()`: Factory function for Neural Network

**`src/train.py`**
- `train_logistic_regression()`: Trains LR models across hyperparameter grid (penalty, C)
- `train_neural_network()`: Trains NN models across hyperparameter grid (learning_rate, optimizer)
- `main()`: Orchestrates data generation, train/test split, and model training
- Logs parameters, metrics, and models to MLflow for each run

**`src/evaluate.py`**
- `get_best_model()`: Queries MLflow to find model with highest test_recall
- Returns best run details, run ID, recall value, and model type

**`src/predict.py`**
- `load_model()`: Loads best model from MLflow based on model type
- `predict()`: Makes predictions on sample input
- Handles both scikit-learn and PyTorch model inference

**`src/serve.py`**
- `save_best_model()`: Saves best model locally for MLflow serving
- Provides command to serve model via `mlflow models serve`

### Usage

1. **Train models**: `python src/train.py`
2. **View experiments**: `mlflow ui`
3. **Find best model**: `python src/evaluate.py`
4. **Make predictions**: `python src/predict.py`
5. **Serve model**: `python src/serve.py` then `mlflow models serve -m best_model -p 5001 --no-conda`

### Design Principles

- **Simple functions**: Each module contains focused, single-purpose functions
- **No error handling**: Prototype-style code focusing on core concepts
- **Modular structure**: Clear separation of concerns (data → models → train → evaluate → predict → serve)
- **Direct MLflow integration**: All scripts use MLflow APIs directly without abstraction layers