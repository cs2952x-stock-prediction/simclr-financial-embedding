import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
import pickle
import torch.nn as nn 

# Custom imports
from util.models import DenseLayers, LstmEncoder, CnnEncoder
from util.datasets import TimeSeriesDataset

#updating the YAML 
print("Check how to change YAML to match checkpoint")
import torch
checkpoint = torch.load("/Users/marcs/Desktop/SSL_final/simclr-financial-embedding/checkpoints/simple-lstm/2024-11-05_031120/encoder_epoch_37.pth")
print(checkpoint.keys())  # Check the keys in the state_dict
# Print the keys and confirm layer shapes
for k, v in checkpoint.items():
    print(f"{k}: {v.shape}")


# Device configuration: Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_checkpoint(checkpoint_path, model):
    """
    Load model weights from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint file.
        model (torch.nn.Module): The model architecture to load the weights into.

    Returns:
        torch.nn.Module: The model with loaded weights, set to evaluation mode.
    """
    state = torch.load(checkpoint_path, map_location=device)
    print({k: state[k].shape for k in state})
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def create_test_dataloader(data_dir, config):
    """
    Create a DataLoader for test data based on the provided directory and configuration.
    The data is already preprocessed (log transformed, differenced, and scaled).

    Args:
        data_dir (str): Path to the directory containing CSV files with test data.
        config (dict): Configuration dictionary specifying data preprocessing parameters.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the time-series test dataset.
    """
    sequences, labels = [], []
    
    filename = "AAPL.csv"
    file_path = os.path.join(data_dir, filename)
    
    if os.path.exists(file_path):
        # Load already preprocessed data
        df = pd.read_csv(file_path)
        print(f"\nProcessing file: {filename}")
        print(f"Shape: {df.shape}")
        
        # Extract features (X) and target variables (Y)
        x_cols = config["data"]["features"]
        y_cols = config["data"]["targets"]
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        sequences.append(df[x_cols].values)
        labels.append(df[y_cols].values)
        print(f"Added sequence of shape: {df[x_cols].values.shape}")

    print(f"\nTotal sequences collected: {len(sequences)}")
    if len(sequences) == 0:
        raise ValueError("No valid sequences were generated")

    dataset = TimeSeriesDataset(
        sequences,
        labels,
        config["data"]["segment_length"],
        config["data"]["segment_step"],
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after creating TimeSeriesDataset")
        
    return DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False)


def fit_linear_regression(X, y):
    """
    Fit a linear regression model to the data and predict the next value.

    Args:
        X (ndarray): Feature matrix (time window of stock prices).
        y (ndarray): Target vector (stock prices).

    Returns:
        ndarray: Predicted next values for each sample.
    """
    predictions = []
    model = LinearRegression()

    for x_seq, y_seq in zip(X, y):
        x_flat = x_seq.reshape(-1, x_seq.shape[-1])  # Flatten the input
        y_seq = y_seq.flatten()

        if len(x_flat) < 2:  # Ensure there's enough data for fitting
            predictions.append(np.nan)
            continue

        # Train on all data
        model.fit(x_flat, y_seq)

        # Predict the next time step using the last available feature vector
        next_x = x_flat[-1].reshape(1, -1)  # Use the last feature vector as input
        next_pred = model.predict(next_x)
        predictions.append(next_pred[0])

    return np.array(predictions)



def generate_visualization(config, output_dir="visualizations"):
    """
    Generate visualizations comparing model predictions vs ground truth.
    """
    # Load scaler for rescaling predictions
    with open(config["data"]["scaler_file"], "rb") as f:
        scaler = pickle.load(f)
    scale = scaler.scale_[-1]  
    mean = scaler.mean_[-1]    

    # Load interim (unprocessed) data for ground truth
    interim_path = "data/interim/kaggle"
    raw_df = pd.read_csv(os.path.join(interim_path, "AAPL.csv"))
    
    # Create a DataLoader for the processed test data
    test_loader = create_test_dataloader(config["data"]["test_dir"], config)

    # Initialize models based on the configuration
    print("Loading models...")

    # Load the LSTM model
    lstm_config = config["models"]["lstm"]
    lstm_encoder = LstmEncoder(
        input_size=lstm_config["input_size"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        proj_size=lstm_config["output_size"],
    ).to(device)
    lstm_probe = DenseLayers(
        input_size=lstm_config["output_size"],
        output_size=1,
        hidden_layers=lstm_config.get("probe_hidden_layers", []),
    ).to(device)
    lstm_encoder = load_model_checkpoint(lstm_config["encoder_checkpoint"], lstm_encoder)
    lstm_probe = load_model_checkpoint(lstm_config["probe_checkpoint"], lstm_probe)
    lstm_model = nn.Sequential(lstm_encoder, lstm_probe)
    lstm_model.eval()

    # Collect predictions
    all_y_pred_linear = []
    all_y_pred_lstm = []

    print("Generating predictions...")
    for x, y in tqdm(test_loader, desc="Generating predictions"):
        # Linear Regression prediction
        y_pred_linear = fit_linear_regression(x.numpy(), y.numpy())

        # LSTM model
        x_seq = x.to(device)
        y_pred_lstm = lstm_model(x_seq)

        # Collect predictions
        all_y_pred_linear.append(y_pred_linear)
        all_y_pred_lstm.append(y_pred_lstm.cpu().detach().numpy())

    # Concatenate all batches
    all_y_pred_linear = np.concatenate(all_y_pred_linear).flatten()
    all_y_pred_lstm = np.concatenate(all_y_pred_lstm).flatten()
    
    print(f"Prediction lengths - Linear: {len(all_y_pred_linear)}, LSTM: {len(all_y_pred_lstm)}")

    def reverse_preprocessing(data, scale, mean, initial_price):
        # 1. Undo scaling
        unscaled = data * scale + mean
        
        # 2. Reconstruct from differences
        prices = np.zeros_like(unscaled)
        prices[0] = np.log(initial_price)
        for i in range(1, len(prices)):
            prices[i] = prices[i-1] + unscaled[i-1]
            
        # 3. Undo log transform
        return np.exp(prices)

    # Get predictions in price space
    raw_prices = raw_df['close'].values[-len(all_y_pred_linear):]  # Match length to predictions
    initial_price = raw_prices[0]  
    y_pred_linear_prices = reverse_preprocessing(all_y_pred_linear, scale, mean, initial_price)
    y_pred_lstm_prices = reverse_preprocessing(all_y_pred_lstm, scale, mean, initial_price)

    print("\nValue Ranges:")
    print(f"Ground Truth (from interim): [{min(raw_prices):.2f}, {max(raw_prices):.2f}]")
    print(f"Linear Regression: [{min(y_pred_linear_prices):.2f}, {max(y_pred_linear_prices):.2f}]")
    print(f"LSTM: [{min(y_pred_lstm_prices):.2f}, {max(y_pred_lstm_prices):.2f}]")

    # Create plot
    plt.figure(figsize=(15, 10))
    sample_range = range(len(raw_prices))
    
    plt.plot(sample_range, raw_prices, 'b-', label="Ground Truth", linewidth=2)
    plt.plot(sample_range, y_pred_linear_prices, 'g--', label="Linear Regression", linewidth=2)
    plt.plot(sample_range, y_pred_lstm_prices, 'r:', label="LSTM Model", linewidth=2)
    
    plt.title("AAPL Stock Price Predictions vs Ground Truth")
    plt.xlabel("Samples")
    plt.ylabel("Stock Price ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # Save the visualization
    output_path = os.path.join(output_dir, "predictions_vs_ground_truth.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations for predictions vs ground truth.")
    parser.add_argument("--config", type=str, required=True, help="Path to the visualization configuration file (YAML format).")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save generated visualizations.")
    args = parser.parse_args()

    # Load configuration from the YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Generate visualizations
    generate_visualization(config, args.output_dir)