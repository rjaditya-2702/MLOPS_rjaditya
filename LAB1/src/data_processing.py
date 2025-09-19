import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """Generate synthetic dataset and save to /data/population_data.csv"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate X values (independent variable)
    # Using a realistic range, let's say from 0 to 100
    n_samples = 1000
    x_min, x_max = 0, 100

    # Generate X with some realistic distribution
    # Mix of uniform and slightly clustered data to mimic real-world scenarios
    x_uniform = np.random.uniform(x_min, x_max, n_samples // 2)
    x_clustered = np.concatenate([
        np.random.normal(30, 10, n_samples // 4),
        np.random.normal(70, 15, n_samples // 4)
    ])
    # Clip clustered values to stay within range
    x_clustered = np.clip(x_clustered, x_min, x_max)

    # Combine and shuffle
    X = np.concatenate([x_uniform, x_clustered])
    np.random.shuffle(X)

    # Generate Y values with the relationship Y = 2X + 3 + noise
    # Add different types of noise to mimic real-world data
    gaussian_noise = np.random.normal(0, 5, n_samples)  # Main noise component
    heteroscedastic_noise = np.random.normal(0, 0.1 * X, n_samples)  # Noise that increases with X
    outlier_mask = np.random.random(n_samples) < 0.02  # 2% outliers
    outlier_noise = np.where(outlier_mask, np.random.normal(0, 30, n_samples), 0)

    # Combine all noise components
    total_noise = gaussian_noise + heteroscedastic_noise * 0.3 + outlier_noise

    # Calculate Y
    Y = 2 * X + 3 + total_noise

    # Create DataFrame
    data = pd.DataFrame({
        'X': X,
        'Y': Y
    })

    # Round to 2 decimal places for cleaner data
    data = data.round(2)

    # Sort by X for easier reading
    data = data.sort_values('X').reset_index(drop=True)

    # Create /data/ directory if it doesn't exist
    os.makedirs(str(Path(__file__).parent.parent / 'data'), exist_ok=True)

    # Save to CSV
    csv_path = Path(__file__).parent.parent / 'data' / 'population_data.csv'
    data.to_csv(csv_path, index=False)

    # print(f"Data generated and saved to {csv_path}")
    # print(f"Number of samples: {n_samples}")
    # print(f"\nFirst 10 rows of the dataset:")
    # print(data.head(10))
    # print(f"\nStatistical summary:")
    # print(data.describe())

    # Calculate actual correlation
    correlation = data['X'].corr(data['Y'])
    # print(f"\nCorrelation between X and Y: {correlation:.4f}")

    # Optional: Create a visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(data['X'], data['Y'], alpha=0.5, s=20)

    # Add the ideal line (without noise)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = 2 * x_line + 3
    plt.plot(x_line, y_line, 'r-', label='Y = 2X + 3 (ideal)', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Data: Y = 2X + 3 + noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_path = Path(__file__).parent.parent / "assets" / "data_distribution.png"
    plt.savefig(plot_path, dpi=100)
    # print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()