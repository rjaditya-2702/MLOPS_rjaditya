"""Save best model locally for serving."""
import mlflow
import os
from evaluate import get_best_model


def save_best_model(output_dir="best_model"):
    """Save the best model locally for serving.
    
    Args:
        output_dir: Directory to save the model
    """
    best_run, best_run_id, best_recall, model_type = get_best_model()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if model_type == "LogisticRegression":
        model_uri = f"runs:/{best_run_id}/LogisticRegressionModel"
        model = mlflow.sklearn.load_model(model_uri)
        mlflow.sklearn.save_model(model, output_dir)
    elif model_type == "NeuralNetwork":
        model_uri = f"runs:/{best_run_id}/NeuralNetworkModel"
        model = mlflow.pytorch.load_model(model_uri)
        mlflow.pytorch.save_model(model, output_dir)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"\nTo serve the model, run:")
    print(f"mlflow models serve -m {os.path.abspath(output_dir)} -p 5001 --no-conda")


if __name__ == "__main__":
    save_best_model()

