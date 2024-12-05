import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
import pickle

# Custom imports
from util.models import DenseLayers, LstmEncoder, CnnEncoder
from util.datasets import TimeSeriesDataset

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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_test_dataloader(data_dir, config):
    """
    Create a DataLoader for test data based on the provided directory and configuration.

    Args:
        data_dir (str): Path to the directory containing CSV files with test data.
        config (dict): Configuration dictionary specifying data preprocessing parameters.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the time-series test dataset.
    """
    sequences, labels = [], []
    for filename in tqdm(os.listdir(data_dir), desc="Loading test data"):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, filename))

            # Extract features (X) and target variables (Y)
            x_cols = config["features"]
            y_cols = config["targets"]
            sequences.append(df[x_cols].values)
            labels.append(df[y_cols].values)

    dataset = TimeSeriesDataset(
        sequences,
        labels,
        config["segment_length"],
        config["segment_step"],
    )
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

def generate_visualization(config, output_dir="visualizations"):
    """
    Generate visualizations comparing model predictions vs ground truth.

    Args:
        config (dict): Experiment configuration dictionary.
        output_dir (str): Directory to save the generated visualizations.
    """
    # Load scaler for rescaling predictions and ground truth
    with open(config["data"]["scaler_file"], "rb") as f:
        scaler = pickle.load(f)
    scale = scaler.scale_[-1]  # Scaling factor for the last target variable
    mean = scaler.mean_[-1]    # Mean value for the last target variable

    # Create a DataLoader for the test data
    test_loader = create_test_dataloader(config["data"]["test_dir"], config["data"])

    # Initialize models based on the configuration
    print("Loading models...")

    # Load the linear model
    linear_config = config["models"]["linear"]
    input_size = linear_config["input_size"]
    linear_model = nn.Linear(input_size, 1)
    linear_checkpoint = linear_config["checkpoint"]
    linear_model = load_model_checkpoint(linear_checkpoint, linear_model)

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

    # Load the CNN model
    cnn_config = config["models"]["cnn"]
    cnn_encoder = CnnEncoder(
        in_channels=cnn_config["in_channels"],
        out_channels=cnn_config["out_channels"],
        kernel_size=cnn_config["kernel_size"],
        num_layers=cnn_config["num_layers"],
    ).to(device)
    cnn_probe = DenseLayers(
        input_size=cnn_config["probe_input_size"],
        output_size=1,
        hidden_layers=cnn_config.get("probe_hidden_layers", []),
    ).to(device)
    cnn_encoder = load_model_checkpoint(cnn_config["encoder_checkpoint"], cnn_encoder)
    cnn_probe = load_model_checkpoint(cnn_config["probe_checkpoint"], cnn_probe)
    cnn_model = nn.Sequential(cnn_encoder, cnn_probe)
    cnn_model.eval()

    # Load the Baseline model
    baseline_config = config["models"]["baseline"]
    baseline_encoder = LstmEncoder(
        input_size=baseline_config["input_size"],
        hidden_size=baseline_config["hidden_size"],
        num_layers=baseline_config["num_layers"],
        proj_size=baseline_config["output_size"],
    ).to(device)
    baseline_probe = DenseLayers(
        input_size=baseline_config["output_size"],
        output_size=1,
        hidden_layers=baseline_config.get("probe_hidden_layers", []),
    ).to(device)
    baseline_checkpoint = baseline_config["checkpoint"]
    baseline_encoder = load_model_checkpoint(baseline_checkpoint, baseline_encoder)
    baseline_model = nn.Sequential(baseline_encoder, baseline_probe)
    baseline_model.eval()

    # Collect predictions and ground truth for visualization
    all_y_true = []
    all_y_pred_linear = []
    all_y_pred_lstm = []
    all_y_pred_cnn = []
    all_y_pred_baseline = []

    print("Generating predictions...")
    for x, y in tqdm(test_loader, desc="Generating predictions"):
        # Ground truth
        y_true = y[:, -1].to(device)

        # Linear model
        x_flat = x.view(x.size(0), -1).to(device)  # Flatten the input
        y_pred_linear = linear_model(x_flat)

        # LSTM model
        x_seq = x.to(device)
        y_pred_lstm = lstm_model(x_seq)

        # CNN model, input shape should be (batch_size, in_channels, seq_len)
        x_cnn = x_seq.permute(0, 2, 1)  # Shape: (batch_size, features, seq_len)
        y_pred_cnn = cnn_model(x_cnn)

        # Baseline model
        y_pred_baseline = baseline_model(x_seq)

        # Rescale predictions and ground truth to original scale
        y_true_rescaled = y_true * scale + mean
        y_pred_linear_rescaled = y_pred_linear * scale + mean
        y_pred_lstm_rescaled = y_pred_lstm * scale + mean
        y_pred_cnn_rescaled = y_pred_cnn * scale + mean
        y_pred_baseline_rescaled = y_pred_baseline * scale + mean

        all_y_true.append(y_true_rescaled.cpu().numpy())
        all_y_pred_linear.append(y_pred_linear_rescaled.cpu().detach().numpy())
        all_y_pred_lstm.append(y_pred_lstm_rescaled.cpu().detach().numpy())
        all_y_pred_cnn.append(y_pred_cnn_rescaled.cpu().detach().numpy())
        all_y_pred_baseline.append(y_pred_baseline_rescaled.cpu().detach().numpy())

    # Concatenate all batches
    all_y_true = np.concatenate(all_y_true).flatten()
    all_y_pred_linear = np.concatenate(all_y_pred_linear).flatten()
    all_y_pred_lstm = np.concatenate(all_y_pred_lstm).flatten()
    all_y_pred_cnn = np.concatenate(all_y_pred_cnn).flatten()
    all_y_pred_baseline = np.concatenate(all_y_pred_baseline).flatten()

    # Plot predictions vs ground truth
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    sample_range = range(len(all_y_true))  # Adjust this if you want to plot a subset
    plt.plot(sample_range, all_y_true, label="Ground Truth", color="blue", alpha=0.7)
    plt.plot(sample_range, all_y_pred_linear, label="Linear Model", color="green", alpha=0.7)
    plt.plot(sample_range, all_y_pred_lstm, label="LSTM Model", color="orange", alpha=0.7)
    plt.plot(sample_range, all_y_pred_cnn, label="CNN Model", color="purple", alpha=0.7)
    plt.plot(sample_range, all_y_pred_baseline, label="Baseline Model", color="red", alpha=0.7)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True)

    # Save the visualization to a file
    output_path = os.path.join(output_dir, "predictions_vs_ground_truth.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

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