# src/ml_pipeline.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import pickle
import json

# Define the neural network architecture
class CircleClassifier(nn.Module):
    def __init__(self):
        super(CircleClassifier, self).__init__()
        self.layer_1 = nn.Linear(2, 16)  # Input layer (2 features: x1, x2) -> 16 neurons
        self.relu = nn.ReLU()            # ReLU activation
        self.layer_2 = nn.Linear(16, 8)  # Hidden layer 1 (16 neurons) -> 8 neurons
        self.relu2 = nn.ReLU()           # ReLU activation
        self.output_layer = nn.Linear(8, 1)  # Output layer (8 neurons) -> 1 neuron
        self.sigmoid = nn.Sigmoid()      # Sigmoid activation for output

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def create_and_save_dataset(output_path='./data/circle_data.csv', **context):
    """
    Task 1: Create a synthetic circle dataset and save it to CSV
    """
    print("Creating synthetic circle dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate synthetic data (circle classification problem)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate points in 2D space
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Points inside the circle (label = 1)
    r_inner = np.random.uniform(0, 1.5, n_samples // 2)
    x1_inner = r_inner * np.cos(theta[:n_samples // 2])
    x2_inner = r_inner * np.sin(theta[:n_samples // 2])
    
    # Points outside the circle (label = 0)
    r_outer = np.random.uniform(2, 3.5, n_samples // 2)
    x1_outer = r_outer * np.cos(theta[n_samples // 2:])
    x2_outer = r_outer * np.sin(theta[n_samples // 2:])
    
    # Combine data
    x1 = np.concatenate([x1_inner, x1_outer])
    x2 = np.concatenate([x2_inner, x2_outer])
    labels = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    
    # Add some noise
    x1 += np.random.normal(0, 0.1, n_samples)
    x2 += np.random.normal(0, 0.1, n_samples)
    
    # Create DataFrame and save
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'label': labels
    })
    
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {data.shape}")
    
    return output_path


def preprocess_data(input_path='./data/circle_data.csv', **context):
    """
    Task 2: Load data and create train/test splits
    """
    print("Loading and preprocessing data...")
    
    # Load the data
    data = pd.read_csv(input_path)
    
    # Split features and labels
    X = data[['x1', 'x2']]
    y = data['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)
    
    # Save preprocessed data for next task
    preprocessed_data = {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'X_train_df': X_train,
        'y_train_df': y_train,
        'X_test_df': X_test,
        'y_test_df': y_test
    }
    
    # Save to pickle file for passing between tasks
    os.makedirs('./temp', exist_ok=True)
    with open('./temp/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"Training set size: {X_train_tensor.shape}")
    print(f"Test set size: {X_test_tensor.shape}")
    
    return './temp/preprocessed_data.pkl'


def train_neural_network(preprocessed_data_path, model_path='./models/circle_classifier.pth', **context):
    """
    Task 3: Train the neural network and save the model
    """
    print("Training neural network...")
    
    # Load preprocessed data
    with open(preprocessed_data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train_tensor = data['X_train']
    y_train_tensor = data['y_train']
    X_test_tensor = data['X_test']
    y_test_tensor = data['y_test']
    
    # Initialize model
    model = CircleClassifier()
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    epochs = 1000
    
    for epoch in range(epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    print("Training finished.")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted_labels = (outputs >= 0.5).float()
        correct_predictions = (predicted_labels == y_test_tensor).sum().item()
        accuracy = correct_predictions / y_test_tensor.size(0)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'epochs': epochs
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Save accuracy to context for logging
    context['task_instance'].xcom_push(key='model_accuracy', value=accuracy)
    
    return model_path


def test_model(model_path='./models/circle_classifier.pth', test_file_path='./test/test.txt', **context):
    """
    Task 4: Load model and make predictions on test.txt file
    """
    print("Testing model with new data...")
    
    # Load the trained model
    model = CircleClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Training accuracy was: {checkpoint['accuracy']:.4f}")
    
    # Read test data from test.txt
    try:
        with open(test_file_path, 'r') as f:
            test_data = f.read().strip()
        
        # Parse comma-separated values
        values = test_data.split(',')
        if len(values) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got {len(values)}")
        
        x1 = float(values[0].strip())
        x2 = float(values[1].strip())

        true_label = float(values[2].strip())
        
        print(f"Test input: x1={x1}, x2={x2}")
        
        # Create tensor and make prediction
        test_input = torch.tensor([[x1, x2]], dtype=torch.float32)
        
        with torch.no_grad():
            output = model(test_input)
            probability = output.item()
            predicted_label = 1 if probability >= 0.5 else 0
        
        # Prepare results
        result = {
            'x1': x1,
            'x2': x2,
            'probability': probability,
            'predicted_label': predicted_label,
            'label_description': 'Inside circle' if predicted_label == 1 else 'Outside circle',
            'true_label': 'Inside circle' if true_label == 1 else 'Outside circle'
        }
        
        # Save results
        os.makedirs('./results', exist_ok=True)
        result_path = './results/prediction_result.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n--- Prediction Results ---")
        print(f"Input: ({x1:.4f}, {x2:.4f})")
        print(f"Probability: {probability:.4f}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Classification: {result['label_description']}")
        print(f"Results saved to {result_path}")
        
        return result
        
    except FileNotFoundError:
        print(f"Error: Test file {test_file_path} not found")
        raise
    except Exception as e:
        print(f"Error processing test file: {str(e)}")
        raise


# Optional: Visualization function (can be called separately)
def visualize_decision_boundary(model_path='./models/circle_classifier.pth', 
                               data_path='./data/circle_data.csv'):
    """
    Optional function to visualize the decision boundary
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load model
    model = CircleClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    data = pd.read_csv(data_path)
    X = data[['x1', 'x2']]
    
    # Create grid for decision boundary
    x1_min, x1_max = X['x1'].min() - 0.5, X['x1'].max() + 0.5
    x2_min, x2_max = X['x2'].min() - 0.5, X['x2'].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                         np.linspace(x2_min, x2_max, 100))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(grid_tensor)
    
    binary_predictions = (predictions >= 0.5).float()
    binary_predictions = binary_predictions.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='x1', y='x2', hue='label', palette='viridis', alpha=0.7)
    plt.contourf(xx, yy, binary_predictions, cmap='coolwarm', alpha=0.5)
    
    plt.title('Learned Decision Boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save plot
    os.makedirs('../../assets', exist_ok=True)
    plt.savefig('../../assets/decision_boundary.png')
    print("Decision boundary plot saved to '../../assets/decision_boundary.png")
    plt.close()