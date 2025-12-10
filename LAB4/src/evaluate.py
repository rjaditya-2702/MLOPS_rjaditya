"""Find the best model based on recall metric."""
from mlflow.tracking import MlflowClient


def get_best_model(experiment_name="Best Binary Classifier"):
    """Get the model with highest recall from MLflow experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        best_run: Best run object
        best_run_id: Best run ID
        best_recall: Best recall value
        model_type: Type of model (LogisticRegression or NeuralNetwork)
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    experiment_id = experiment.experiment_id
    
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="",
        order_by=["metrics.test_recall DESC"]
    )
    
    best_run = runs[0]
    best_recall = best_run.data.metrics.get("test_recall", 0)
    best_run_id = best_run.info.run_id
    
    model_type = None
    for tag_key, tag_value in best_run.data.tags.items():
        if tag_key == "model_type":
            model_type = tag_value
            break
    
    print(f"Best Run ID: {best_run_id}")
    print(f"Highest recall: {best_recall:.4f}")
    print(f"Model type: {model_type}")
    print("\nBest Model Parameters:")
    for param_key, param_value in best_run.data.params.items():
        print(f"  {param_key}: {param_value}")
    
    return best_run, best_run_id, best_recall, model_type


if __name__ == "__main__":
    get_best_model()

