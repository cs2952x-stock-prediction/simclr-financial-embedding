import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TimeSeriesDataset
from evaluation import nt_xent_loss
from models import SimpleLstmEncoder
from transformations import mask_with_added_gaussian

# Configurations
embedding_dim = 64  # Dimension of embeddings produced by the model
hidden_dim = 128  # Hidden LSTM dimension
output_dim = 64  # Output projection dimension (for SimCLR projection head)
batch_size = 32  # Number of samples per batch
temperature = 0.5  # Temperature parameter for NT-Xent loss
n_epochs = 100  # Number of epochs

segment_length = 14  # on daily entries, this is two weeks

data_file = "data/raw/kaggle_sp500/sp500_stocks.csv"


def train_one_epoch(model, data_loader, optimizer, temperature):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        # Apply two augmentations to the batch
        x_i = mask_with_added_gaussian(batch.numpy(), mask_prob=0.1, std_multiplier=0.1)
        x_j = batch.numpy()  # NOTE: currently just the identity function

        # Convert augmented data to tensors and move to device
        x_i = torch.tensor(x_i, dtype=torch.float32).to(device)
        x_j = torch.tensor(x_j, dtype=torch.float32).to(device)

        # Forward pass for both views
        z_i = model(x_i)
        z_j = model(x_j)

        # Concatenate views and compute contrastive loss
        z = torch.stack([z_i, z_j], dim=0)
        loss = nt_xent_loss(z, temperature)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


if __name__ == "__main__":
    # Load data
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"Loaded data with shape {df.shape}")
    print(df.head())

    # Prepare data for training
    print("Preparing data for training...")

    # Convert datetime to numeric
    print("Converting datetime to numeric...")
    datetime = pd.to_datetime(df["datetime"])
    df.drop(columns=["datetime"], inplace=True)
    df["time"] = (datetime - datetime.min()).dt.total_seconds()

    # Separate series by symbol
    print("Separating series by symbol...")
    series = []
    symbols = df["symbol"].unique()
    for symbols in tqdm(symbols):
        symbol_series = df[df["symbol"] == symbols]
        symbol_series = symbol_series.drop(columns=["symbol"])
        # Interpolate missing values
        symbol_series = symbol_series.interpolate(method="linear", axis=0)
        series.append(symbol_series.values)

    print("Creating dataset and data loader...")
    dataset = TimeSeriesDataset(series, segment_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLstmEncoder(
        n_features=series[0].shape[-1],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training loop...")
    for epoch in range(n_epochs):
        avg_loss = train_one_epoch(model, data_loader, optimizer, temperature)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
