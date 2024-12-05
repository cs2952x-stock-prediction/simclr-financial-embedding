import os
import logging
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
from datetime import datetime
from util.training import linear_epoch

# Custom imports (adjust the paths based on your project structure)
from util.datasets import TimeSeriesDataset

############################# GLOBAL VARIABLES #####################################################
# WandB API key
WANDB_KEY = os.getenv("WANDB_API_KEY")
assert WANDB_KEY is not None, "WANDB_API_KEY environment variable not found."
os.environ["WANDB_API_KEY"] = WANDB_KEY

# Logger setup
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"logs/simple-linear_{timestamp}.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint directory
CHECKPOINTS_DIR = f"checkpoints/simple-linear/{timestamp}"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

############################# FUNCTIONS ############################################################


def build_dataloader(data_dir, batch_size, segment_length, segment_step, features, target):
    """
    Build a DataLoader for time-series data.
    """
    sequences, labels = [], []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, filename))
            sequences.append(df[features].values)
            labels.append(df[target].values)

    dataset = TimeSeriesDataset(sequences, labels, segment_length, segment_step)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def initialize_model(input_size, output_size):
    """
    Initialize a simple linear regression model.
    """
    return nn.Linear(input_size, output_size).to(device)


# def save_checkpoint(model, optimizer, epoch):
#     """
#     Save the model and optimizer checkpoints.
#     """
#     checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch + 1:03d}")
#     os.makedirs(checkpoint_path, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pth"))
#     torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pth"))
#     logger.info(f"Model checkpoint saved at {checkpoint_path}")

def save_checkpoint(model, optimizer, epoch, config=None):
    """
    Save the model, optimizer, and additional metadata (e.g., config, epoch).
    """
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch + 1:03d}")
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,  # Optional: Save configuration for reproducibility
    }, os.path.join(checkpoint_path, "checkpoint.pth"))
    logger.info(f"Checkpoint saved at {checkpoint_path}")


############################# MAIN FUNCTION #######################################################


def main():
    # Hyperparameters and data paths
    train_dir = "data/processed/kaggle/v2/train"
    test_dir = "data/processed/kaggle/v2/test"
    batch_size = 64
    segment_length = 30
    segment_step = 5
    features = ["open", "close", "high", "low", "volume"]
    target = ["close"]
    learning_rate = 0.001
    num_epochs = 10

    # Initialize WandB
    wandb.init(project="simcl-stock-embedding", name="simple-linear", config={
        "batch_size": batch_size,
        "segment_length": segment_length,
        "segment_step": segment_step,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    })

    # Load data
    logger.info("Loading data...")
    train_loader = build_dataloader(train_dir, batch_size, segment_length, segment_step, features, target)
    test_loader = build_dataloader(test_dir, batch_size, segment_length, segment_step, features, target)

    # Model setup
    input_size = segment_length * len(features)  # Updated input size
    output_size = 1  # Assuming regression on a single target variable
    model = initialize_model(input_size, output_size)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train model
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x_flat = x.view(x.size(0), -1).to(device)  # Flatten the input
            y_pred = model(x_flat)
            y_true = y[:, -1].to(device)  # Target is the last time step
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        logger.info(f"Training Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

        # Test model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x_flat = x.view(x.size(0), -1).to(device)
                y_pred = model(x_flat)
                y_true = y[:, -1].to(device)
                loss = criterion(y_pred, y_true)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "test_loss": test_loss})

        # Save checkpoints
        save_checkpoint(model, optimizer, epoch)

    wandb.finish()
    logger.info("Training completed.")


if __name__ == "__main__":
    main()