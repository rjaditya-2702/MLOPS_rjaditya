"""Load best model and make predictions."""
import mlflow
import numpy as np
import torch
from evaluate import get_best_model


def load_model(best_run_id, model_type):
    """Load model from MLflow run.
    
    Args:
        best_run_id: MLflow run ID
        model_type: Type of model (LogisticRegression or NeuralNetwork)
        
    Returns:
        model: Loaded model
    """
    if model_type == "LogisticRegression":
        model_path = f"runs:/{best_run_id}/LogisticRegressionModel"
        model = mlflow.sklearn.load_model(model_path)
    elif model_type == "NeuralNetwork":
        model_path = f"runs:/{best_run_id}/NeuralNetworkModel"
        model = mlflow.pytorch.load_model(model_path)
    
    return model


def predict(model, model_type, sample_input):
    """Make predictions using the loaded model.
    
    Args:
        model: Trained model
        model_type: Type of model (LogisticRegression or NeuralNetwork)
        sample_input: Input data for prediction
        
    Returns:
        predictions: Model predictions
    """
    if model_type == "NeuralNetwork":
        sample_input_tensor = torch.tensor(sample_input, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input_tensor)
            predictions = (outputs > 0.5).int().numpy()
    else:
        predictions = model.predict(sample_input)
    
    return predictions


def main():
    """Main prediction function."""
    best_run, best_run_id, best_recall, model_type = get_best_model()
    
    model = load_model(best_run_id, model_type)
    print(f"\nLoaded {model_type} model")
    
    sample_input = np.array([
        [0.5, 0.5],
        [-0.5, 0.8],
        [1.2, -0.5],
        [-1.0, -1.0]
    ])
    
    predictions = predict(model, model_type, sample_input)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        if model_type == "NeuralNetwork":
            print(f"  Sample {i+1}: X = {sample_input[i]} || Prediction: {pred[0]}")
        else:
            print(f"  Sample {i+1}: X = {sample_input[i]} || Prediction: {pred}")


if __name__ == "__main__":
    main()

