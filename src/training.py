import os

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
from models import DenseLayers, LstmEncoder
from transformations import mask_with_added_gaussian

# Run configuration

config = {
    # emcoding model config
    "hidden_size": 128,  # LSTM c_n
    "representation_size": 64,  # LSTM h_n (output size of encoder)
    "num_layers": 1,
    # projection head config
    "proj_layers": [(64, "relu"), (32, "relu")],
    "proj_size": 16,  # output size of projection head
    # linear probe config
    "probe_layers": [(64, "relu"), (32, "relu")],
    "probe_size": 1,
    # training config
    "learning_rate": 1e-6,
    "batch_size": 128,
    "n_epochs": 100,
    "temperature": 0.5,
    # data source+structure config
    "data": "data/interim/kaggle_sp500",
    "segment_length": 30,
    "segment_step": 3,
}

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

wandb.init(project="simcl-stock-embedding", name="simple-lstm-encoder", config=config)


def train_one_epoch(encoder, projector, data_loader, optimizer, temperature):
    encoder.train()
    projector.train()
    total_loss = 0

    pbar = tqdm(data_loader)
    for batch in pbar:
        # Apply two augmentations to the batch
        x_i = mask_with_added_gaussian(batch.numpy(), mask_prob=1.0, std_multiplier=0.2)
        x_j = batch.numpy()  # NOTE: currently just the identity function

        # Convert augmented data to tensors and move to device
        x_i = torch.tensor(x_i, dtype=torch.float32).to(device)
        x_j = torch.tensor(x_j, dtype=torch.float32).to(device)

        # Forward pass through the encoder
        h_i = encoder(x_i)
        h_j = encoder(x_j)

        # Forward pass through the projection head
        z_i = projector(h_i)
        z_j = projector(h_j)

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
    data_dir = config["data"]
    print(f"Loading data from {data_dir}")

    series = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(config["data"], filename))
            datetime = pd.to_datetime(df["datetime"])
            df.drop(columns=["datetime"], inplace=True)
            df["time"] = (datetime - datetime.min()).dt.total_seconds()
            series.append(df.values)

    print("Creating dataset and data loader...")
    dataset = TimeSeriesDataset(
        series,
        config["segment_length"],
        step=config["segment_step"],
    )
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model and optimizer
    print("Initializing model and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LstmEncoder(
        input_size=series[0].shape[-1],
        hidden_size=config["hidden_size"],
        proj_size=config["representation_size"],
    ).to(device)
    projector = DenseLayers(
        input_size=config["representation_size"],
        output_size=config["proj_size"],
        intermediates=config["proj_layers"],
    ).to(device)
    model_params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = Adam(model_params, lr=config["learning_rate"])

    # Model summary
    print("Encoder model summary:")
    summary(
        encoder,
        input_data=(
            config["segment_length"],
            series[0].shape[-1],
        ),
    )

    print("Projector model summary:")
    summary(
        projector,
        input_data=(config["representation_size"],),
    )

    # Training loop
    print("Starting training loop...")
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):
        epoch_loss = train_one_epoch(
            encoder, projector, data_loader, optimizer, config["temperature"]
        )
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")
        wandb.log({"Epoch": epoch, "Epoch Loss": epoch_loss})

    wandb.finish()
