import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import wandb
from datasets import TimeSeriesDataset
from evaluation import nt_xent_loss
from models import SimpleLstmEncoder
from transformations import mask_with_added_gaussian

# Configurations

hidden_size = 128  # Dimension of lstm hidden cell state
proj_size = 0  # Dimension of lstm output
output_size = 32  # Output projection dimension (for SimCLR projection head)

learning_rate = 1e-6  # Learning rate for the optimizer
batch_size = 128  # Number of samples per batch
n_epochs = 100  # Number of epochs
temperature = 0.5  # Temperature parameter for NT-Xent loss

data_file = "data/raw/kaggle_sp500/sp500_stocks.csv"
segment_length = 30  # on daily entries, this is one month
segment_step = 10  # the distance between the start of each segment

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

wandb.init(
    project="simcl-stock-embedding",
    name="simple-lstm-encoder",
    config={
        # model config
        "hidden_size": hidden_size,
        "proj_size": proj_size,
        "output_size": output_size,
        # training config
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "temperature": temperature,
        # data source+structure config
        "data_file": data_file,
        "segment_length": segment_length,
        "segment_step": segment_step,
    },
)


def train_one_epoch(model, data_loader, optimizer, temperature):
    model.train()
    total_loss = 0

    pbar = tqdm(data_loader)
    for batch in pbar:
        # Apply two augmentations to the batch
        x_i = mask_with_added_gaussian(batch.numpy(), mask_prob=1.0, std_multiplier=0.2)
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

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(data_loader)
    return avg_loss


if __name__ == "__main__":
    # Load data
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(df.head())

    # Prepare data for training
    print("Preparing data for training...")

    # Convert datetime to numeric
    print("Converting datetime to numeric...")
    datetime = pd.to_datetime(df["datetime"])
    df.drop(columns=["datetime"], inplace=True)
    df["time"] = (datetime - datetime.min()).dt.total_seconds()

    # Separate series by symbol
    print("Separating series by symbol, interpolating, removing nan values...")
    series = []
    symbols = df["symbol"].unique()
    for symbol in tqdm(symbols):
        symbol_series = df[df["symbol"] == symbol]
        symbol_series = symbol_series.drop(columns=["symbol"])

        # Interpolate missing values
        symbol_series = symbol_series.interpolate(method="linear", axis=0)
        symbol_series.dropna(inplace=True)
        series.append(symbol_series.values)

    print("Creating dataset and data loader...")
    dataset = TimeSeriesDataset(
        series,
        segment_length,
        step=segment_step,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLstmEncoder(
        input_size=series[0].shape[-1],
        hidden_size=hidden_size,
        output_size=output_size,
        proj_size=proj_size,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    summary(
        model,
        input_data=(segment_length, series[0].shape[-1]),
    )

    # Training loop
    print("Starting training loop...")
    for epoch in range(n_epochs):
        avg_loss = train_one_epoch(model, data_loader, optimizer, temperature)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        wandb.log({"Epoch": epoch, "Epoch Loss": avg_loss})

    wandb.finish()
