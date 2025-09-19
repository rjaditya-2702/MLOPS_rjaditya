import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def main():
    # Load the data
    data_path = Path(__file__).parent.parent / 'data' / 'population_data.csv'
    data = pd.read_csv(data_path)

    print("=" * 60)
    print("LINEAR REGRESSION MODEL FOR Y = 2X + 3 + noise")
    print("=" * 60)
    print(f"\nDataset loaded from: {data_path}")
    print(f"Shape of dataset: {data.shape}")

    # Prepare features and target
    X = data[['X']].values  # Features (need 2D array for sklearn)
    y = data['Y'].values    # Target

    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # print(f"\nTraining set size: {len(X_train)} samples")
    # print(f"Testing set size: {len(X_test)} samples")

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get model parameters
    slope = model.coef_[0]
    intercept = model.intercept_


    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Training':<15} {'Testing':<15}")
    print("-" * 55)
    print(f"{'Mean Squared Error (MSE)':<25} {train_mse:<15.4f} {test_mse:<15.4f}")
    print(f"{'Root MSE (RMSE)':<25} {np.sqrt(train_mse):<15.4f} {np.sqrt(test_mse):<15.4f}")
    print(f"{'Mean Absolute Error (MAE)':<25} {train_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'R² Score':<25} {train_r2:<15.4f} {test_r2:<15.4f}")

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    print(f"\n{'Residuals Statistics':<25} {'Training':<15} {'Testing':<15}")
    print("-" * 55)
    print(f"{'Mean of Residuals':<25} {np.mean(train_residuals):<15.4f} {np.mean(test_residuals):<15.4f}")
    print(f"{'Std of Residuals':<25} {np.std(train_residuals):<15.4f} {np.std(test_residuals):<15.4f}")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(X_train, y_train, alpha=0.5, label='Training data', s=20)
    ax1.scatter(X_test, y_test, alpha=0.5, label='Testing data', s=20)
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line_pred = model.predict(x_line)
    ax1.plot(x_line, y_line_pred, 'r-', linewidth=2, label=f'Fitted: Y={slope:.2f}X+{intercept:.2f}')
    ax1.plot(x_line, 2*x_line + 3, 'g--', linewidth=2, alpha=0.7, label='True: Y=2X+3')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Linear Regression Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Predicted vs Actual (Training)
    ax2 = axes[0, 1]
    ax2.scatter(y_train, y_train_pred, alpha=0.5, s=20)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Y (Training)')
    ax2.set_ylabel('Predicted Y')
    ax2.set_title(f'Training: Predicted vs Actual\nR² = {train_r2:.4f}')
    ax2.grid(True, alpha=0.3)

    # 3. Predicted vs Actual (Testing)
    ax3 = axes[0, 2]
    ax3.scatter(y_test, y_test_pred, alpha=0.5, s=20, color='orange')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('Actual Y (Testing)')
    ax3.set_ylabel('Predicted Y')
    ax3.set_title(f'Testing: Predicted vs Actual\nR² = {test_r2:.4f}')
    ax3.grid(True, alpha=0.3)

    # 4. Residuals vs Predicted (Training)
    ax4 = axes[1, 0]
    ax4.scatter(y_train_pred, train_residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Y (Training)')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Training: Residual Plot')
    ax4.grid(True, alpha=0.3)

    # 5. Residuals vs Predicted (Testing)
    ax5 = axes[1, 1]
    ax5.scatter(y_test_pred, test_residuals, alpha=0.5, s=20, color='orange')
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicted Y (Testing)')
    ax5.set_ylabel('Residuals')
    ax5.set_title('Testing: Residual Plot')
    ax5.grid(True, alpha=0.3)

    # 6. Distribution of Residuals
    ax6 = axes[1, 2]
    ax6.hist(train_residuals, bins=30, alpha=0.5, label='Training', density=True)
    ax6.hist(test_residuals, bins=30, alpha=0.5, label='Testing', density=True)
    ax6.set_xlabel('Residuals')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution of Residuals')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Linear Regression Model Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the plot
    plot_path = Path(__file__).parent.parent / "assets" / "model_analysis.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"\n" + "=" * 60)
    print(f"Visualization saved to: {plot_path}")

    # Make predictions for new data points
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    sample_X = np.array([[10], [25], [50], [75], [100]])
    sample_predictions = model.predict(sample_X)
    print(f"\n{'X Value':<15} {'Predicted Y':<15} {'True Y (no noise)':<20}")
    print("-" * 50)
    for x_val, y_pred in zip(sample_X.flatten(), sample_predictions):
        true_y = 2 * x_val + 3
        print(f"{x_val:<15.1f} {y_pred:<15.2f} {true_y:<20.1f}")

    # Calculate confidence intervals (approximate)
    # Standard error of predictions
    se = np.sqrt(test_mse)
    confidence_level = 0.95
    z_score = 1.96  # for 95% confidence

    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    print(f"Approximate prediction interval: ± {z_score * se:.2f}")
    print(f"This means most predictions should fall within ± {z_score * se:.2f} of the predicted value")

    # Save the model parameters to a file
    model_info = {
        'slope': slope,
        'intercept': intercept,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': np.sqrt(train_mse),
        'test_rmse': np.sqrt(test_mse)
    }

    model_df = pd.DataFrame([model_info])
    model_df.to_csv(Path(__file__).parent.parent / 'model' / 'model_parameters.csv', index=False)
    # print(f"\nModel parameters saved to: /data/model_parameters.csv")

if __name__ == "__main__":
    main()