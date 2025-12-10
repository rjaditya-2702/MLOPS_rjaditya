"""Model definitions for Logistic Regression and Neural Network."""
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


def create_lr_model(penalty='l2', C=1.0, random_state=42):
    """Create Logistic Regression model.
    
    Args:
        penalty: Regularization penalty ('l1' or 'l2')
        C: Inverse of regularization strength
        random_state: Random seed
        
    Returns:
        LogisticRegression model
    """
    return LogisticRegression(
        penalty=penalty,
        C=C,
        solver='liblinear',
        random_state=random_state
    )


class SimpleNN(nn.Module):
    """Simple PyTorch Neural Network for binary classification."""
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input layer (2 features) to hidden layer (16 neurons)
        self.relu = nn.ReLU()        # Activation function
        self.fc2 = nn.Linear(16, 1)  # Hidden layer to output layer (1 neuron)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def create_nn_model():
    """Create PyTorch Neural Network model.
    
    Returns:
        SimpleNN model
    """
    return SimpleNN()

