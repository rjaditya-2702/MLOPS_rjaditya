"""Generate synthetic binary classification dataset."""
from sklearn.datasets import make_classification


def generate_data(n_samples=1000, n_features=2, random_state=42):
    """Generate synthetic binary classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=random_state
    )
    return X, y

