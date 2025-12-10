"""Train Logistic Regression and Neural Network models with hyperparameter search."""
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
import itertools
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data import generate_data
from models import create_lr_model, create_nn_model


def train_logistic_regression(X_train, X_test, y_train, y_test, experiment_name):
    """Train Logistic Regression models with different hyperparameters.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    
    lr_param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10, 100],
    }
    
    for lr_params in itertools.product(*lr_param_grid.values()):
        lr_param_dict = dict(zip(lr_param_grid.keys(), lr_params))
        run_name = f"LR_penalty{lr_param_dict['penalty']}_C{lr_param_dict['C']}"
        
        with mlflow.start_run(nested=True, run_name=run_name):
            mlflow.set_tags({
                "model_type": "LogisticRegression",
                "regularization": lr_param_dict['penalty'],
            })
            
            mlflow.log_params(lr_param_dict)
            
            lr = create_lr_model(**lr_param_dict)
            lr.fit(X_train, y_train)
            
            y_pred_lr = lr.predict(X_test)
            y_pred_train = lr.predict(X_train)
            signature = infer_signature(X_train, y_pred_train)
            
            accuracy_lr = accuracy_score(y_test, y_pred_lr)
            precision_lr = precision_score(y_test, y_pred_lr, zero_division=1)
            recall_lr = recall_score(y_test, y_pred_lr)
            
            mlflow.log_metric("test_accuracy", accuracy_lr)
            mlflow.log_metric("test_precision", precision_lr)
            mlflow.log_metric("test_recall", recall_lr)
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    lr, "LogisticRegressionModel", 
                    registered_model_name=run_name, 
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(lr, "LogisticRegressionModel", signature=signature)


def train_neural_network(X_train, X_test, y_train, y_test, experiment_name):
    """Train Neural Network models with different hyperparameters.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(log_datasets=True)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    nn_param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'optimizer': ['Adam', 'SGD'],
    }
    
    for nn_params in itertools.product(*nn_param_grid.values()):
        nn_param_dict = dict(zip(nn_param_grid.keys(), nn_params))
        run_name = f"NN_{nn_param_dict['optimizer']}_lr{nn_param_dict['learning_rate']}"
        
        with mlflow.start_run(nested=True, run_name=run_name):
            mlflow.set_tags({
                "model_type": "NeuralNetwork",
                "optimizer": nn_param_dict['optimizer'],
                "loss_function": 'BCELoss'
            })
            
            nn_model = create_nn_model()
            learning_rate = nn_param_dict['learning_rate']
            optimizer_type = nn_param_dict['optimizer']
            
            if optimizer_type == 'Adam':
                optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)
            elif optimizer_type == 'SGD':
                optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)
            
            criterion = torch.nn.BCELoss()
            epochs = 12
            
            for epoch in range(epochs):
                nn_model.train()
                epoch_loss = 0.0
                batches = 0
                
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = nn_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batches += 1
                
                avg_loss = epoch_loss / batches
                mlflow.log_metric("training_loss", avg_loss, step=epoch)
            
            nn_model.eval()
            with torch.no_grad():
                y_pred_tensor_nn = nn_model(X_test_tensor)
                y_pred_train = nn_model(X_train_tensor)
                y_pred_train_numpy = y_pred_train.numpy()
                y_pred_nn = (y_pred_tensor_nn > 0.5).int().numpy()
            
            signature = infer_signature(
                X_train_tensor.numpy(), 
                y_pred_train_numpy
            )
            
            mlflow.log_params(nn_param_dict)
            
            accuracy_nn = accuracy_score(y_test_tensor, y_pred_nn)
            precision_nn = precision_score(y_test_tensor, y_pred_nn, zero_division=1)
            recall_nn = recall_score(y_test_tensor, y_pred_nn)
            
            mlflow.log_metric("test_accuracy", accuracy_nn)
            mlflow.log_metric("test_precision", precision_nn)
            mlflow.log_metric("test_recall", recall_nn)
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    nn_model, 
                    "NeuralNetworkModel", 
                    registered_model_name=run_name,
                    signature=signature
                )
            else:
                mlflow.pytorch.log_model(
                    nn_model, 
                    "NeuralNetworkModel", 
                    signature=signature
                )
            
            mlflow.log_dict(
                {
                    "dataset_info": {
                        "training_samples": X_train.shape[0],
                        "test_samples": X_test.shape[0], 
                        "features": X_train.shape[1],
                        "class_distribution": np.bincount(y_train).tolist()
                    }
                }, 
                "dataset_info.json"
            )


def main():
    """Main training function."""
    mlflow.autolog()
    
    experiment_name = "Best Binary Classifier"
    
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Logistic Regression models...")
    train_logistic_regression(X_train, X_test, y_train, y_test, experiment_name)
    
    print("Training Neural Network models...")
    train_neural_network(X_train, X_test, y_train, y_test, experiment_name)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

