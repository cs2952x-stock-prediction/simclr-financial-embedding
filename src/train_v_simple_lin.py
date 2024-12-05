import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAIN_DIR = "data/processed/kaggle/v2/train"
TEST_DIR = "data/processed/kaggle/v2/test"
SEGMENT_LENGTH = 30
SEGMENT_STEP = 5
FEATURES = ["open", "close", "high", "low", "volume"]
TARGET = "close"
BATCH_SIZE = 64

# Function to load data
def load_data(data_dir, features, target, segment_length, segment_step):
    """
    Load time-series data from a directory and create sequences.

    Args:
    - data_dir: Directory containing CSV files.
    - features: List of features to extract.
    - target: Target column to predict.
    - segment_length: Length of each time segment.
    - segment_step: Step size between segments.

    Returns:
    - X: List of feature sequences.
    - y: List of target values.
    """
    sequences, targets = [], []
    for filename in tqdm(os.listdir(data_dir), desc="Loading data"):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, filename))

            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[features])
            scaled_df = pd.DataFrame(scaled_data, columns=features)

            # Create sliding window segments
            for i in range(0, len(scaled_df) - segment_length, segment_step):
                segment = scaled_df.iloc[i:i+segment_length].values
                next_value = df[target].iloc[i + segment_length]  # True next value
                sequences.append(segment)
                targets.append(next_value)

    return np.array(sequences), np.array(targets)

# Fit linear regression and predict
def fit_and_predict(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model and make predictions.

    Args:
    - X_train: Training feature sequences.
    - y_train: Training target values.
    - X_test: Test feature sequences.
    - y_test: Test target values.

    Returns:
    - predictions: Predicted values for the test set.
    """
    predictions = []
    model = LinearRegression()

    for i, x in tqdm(enumerate(X_test), desc="Making predictions", total=len(X_test)):
        # Flatten the input for linear regression
        x_train_flat = X_train.reshape(X_train.shape[0], -1)
        x_test_flat = x.reshape(1, -1)  # Test sample

        # Fit the model on the training data
        model.fit(x_train_flat, y_train)

        # Predict the next stock price
        next_value = model.predict(x_test_flat)
        predictions.append(next_value[0])

    return predictions

# Evaluate the model
def evaluate(y_true, y_pred):
    """
    Evaluate predictions using RMSE and MAE.

    Args:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")

    return rmse, mae

# Visualize results
def plot_results(y_true, y_pred):
    """
    Plot predictions vs ground truth.

    Args:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Values", color="blue", alpha=0.7)
    plt.plot(y_pred, label="Predictions", color="orange", alpha=0.7)
    plt.legend()
    plt.title("Linear Regression Predictions vs Ground Truth")
    plt.xlabel("Samples")
    plt.ylabel("Stock Price")
    plt.show()

def main():
    # Load training and testing data
    logger.info("Loading training data...")
    X_train, y_train = load_data(TRAIN_DIR, FEATURES, TARGET, SEGMENT_LENGTH, SEGMENT_STEP)

    logger.info("Loading testing data...")
    X_test, y_test = load_data(TEST_DIR, FEATURES, TARGET, SEGMENT_LENGTH, SEGMENT_STEP)

    # Fit model and make predictions
    logger.info("Training and predicting...")
    predictions = fit_and_predict(X_train, y_train, X_test, y_test)

    # Evaluate results
    logger.info("Evaluating predictions...")
    evaluate(y_test, predictions)

    # Visualize predictions
    logger.info("Visualizing results...")
    plot_results(y_test, predictions)

if __name__ == "__main__":
    main()