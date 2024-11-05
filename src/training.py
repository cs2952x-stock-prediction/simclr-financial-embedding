import os
from copy import deepcopy

import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import wandb
from datasets import TimeSeriesDataset
from evaluation import nt_xent_loss
from models import DenseLayers, LstmEncoder
from transformations import mask_with_added_gaussian


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def train_batch_simclr(x, transform, model, temp, optimizer):
    # transform data
    x_i = transform(x)
    x_j = transform(x)

    # forward pass
    z_i = model(x_i)
    z_j = model(x_j)

    loss = nt_xent_loss(z_i, z_j, temp)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_batch(x, y, model, loss, optimizer):
    y_pred = model(x).squeeze()
    loss_val = loss(y_pred, y)

    # backward pass
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    return loss_val


def training_run(
    encoder,
    projector,
    probe,
    baseline_model,
    train_loader,
    test_loader,
    simclr_optimizer,
    probe_optimizer,
    baseline_optimizer,
    n_epochs,
):
    pretext_model = nn.Sequential(encoder, projector)
    task_model = nn.Sequential(encoder, probe)

    # SimCLR to train encoder
    transform = lambda x: mask_with_added_gaussian(
        x, mask_prob=0.1, std_multiplier=0.01
    )

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print("Training...")
        total_simclr_loss = 0
        total_task_loss = 0
        total_baseline_loss = 0
        pretext_model.train()
        task_model.train()
        baseline_model.train()
        pbar = tqdm(train_loader)
        for x, y in pbar:
            simclr_loss = train_batch_simclr(
                x,
                transform,
                pretext_model,
                config["temperature"],
                simclr_optimizer,
            )

            # Probe training
            freeze(encoder)
            task_loss = train_batch(
                x, y[:, -1], task_model, nn.MSELoss(), probe_optimizer
            )
            unfreeze(encoder)

            # Supervised training
            baseline_loss = train_batch(
                x, y[:, -1], baseline_model, nn.MSELoss(), baseline_optimizer
            )

            pbar.set_postfix(
                {
                    "SimCLR Loss": f"{simclr_loss.item():.4e}",
                    "Task Loss": f"{task_loss.item():.4e}",
                    "Baseline Loss": f"{baseline_loss.item():.4e}",
                }
            )

            wandb.log(
                {
                    "SimCLR Batch Loss": simclr_loss.item(),
                    "Task Batch Loss": task_loss.item(),
                    "Baseline Batch Loss": baseline_loss.item(),
                }
            )

            total_simclr_loss += simclr_loss.item()
            total_task_loss += task_loss.item()
            total_baseline_loss += baseline_loss.item()

        avg_simclr_loss = total_simclr_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        avg_baseline_loss = total_baseline_loss / len(train_loader)

        print(f"SimCLR Training Loss: {avg_simclr_loss:.4e}")
        print(f"Task Training Loss: {avg_task_loss:.4e}")
        print(f"Supervised Training Loss: {avg_baseline_loss:.4e}")

        print("Evaluating...")
        task_val_loss = 0
        baseline_val_loss = 0
        pretext_model.eval()
        task_model.eval()
        pbar = tqdm(test_loader)
        for x, y in pbar:
            # Probe training
            y_pred = task_model(x).squeeze()
            task_val_loss += nn.MSELoss()(y_pred, y[:, -1]).item()

            # Supervised training
            y_pred = baseline_model(x).squeeze()
            baseline_val_loss += nn.MSELoss()(y_pred, y[:, -1]).item()

        avg_task_val_loss = task_val_loss / len(test_loader)
        avg_baseline_val_loss = baseline_val_loss / len(test_loader)

        print(f"Avg Task Validation Loss: {avg_task_val_loss:.4e}")
        print(f"Avg Supervised Validation Loss: {avg_baseline_val_loss:.4e}")

        wandb.log(
            {
                "Epoch": epoch + 1,
                "SimCLR Training Loss": avg_simclr_loss,
                "Task Training Loss": avg_task_loss,
                "Supervised Training Loss": avg_baseline_loss,
                "Task Validation Loss": avg_task_val_loss,
                "Supervised Validation Loss": avg_baseline_val_loss,
            }
        )


if __name__ == "__main__":
    # Run configuration
    config = {
        # encoding model config
        "hidden_size": 128,  # LSTM c_n (hidden cell size)
        "representation_size": 128,  # LSTM h_n (output size of encoder) NOTE: setting c_n = h_n enables torch Dnn optimization
        "num_layers": 1,
        # projection head config
        "proj_layers": [(64, "relu"), (32, "relu")],
        "proj_size": 16,  # output size of projection head
        # linear probe config
        "probe_layers": [(64, "relu"), (32, "relu")],
        "probe_size": 1,
        # training config
        "simclr_lr": 1e-5,  # learning rate for encoder and projector during SimCLR training
        "probe_lr": 1e-1,  # learning rate for linear probe appended to encoder trained with SimCLR
        "baseline_lr": 1e-1,  # learning rate for baseline supervised model
        "batch_size": 128,
        "n_epochs": 100,
        "temperature": 0.5,
        # data source+structure config
        "data": "data/interim/kaggle_sp500",
        "train_cutoff_date": pd.Timestamp("2023-01-01"),
        "segment_length": 30,
        "segment_step": 3,
    }

    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key is None:
        raise ValueError("WANDB_API_KEY environment variable no found")
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(
        project="simcl-stock-embedding", name="simple-lstm-encoder", config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = config["data"]
    print(f"Loading data from {data_dir}")
    print(f"All data prior to {config['train_cutoff_date']} will be used for training.")

    train_seqs = []
    test_seqs = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(config["data"], filename))
            df.drop(
                columns=["adj_close"], inplace=True
            )  # TODO: for now, toss the adjusted close column

            column_order = [col for col in df.columns if col != "close"] + [
                "close"
            ]  # TODO: for now, move close column to end (last column is label)
            df = pd.DataFrame(df[column_order])
            datetime = pd.to_datetime(df["datetime"])
            df.drop(columns=["datetime"], inplace=True)
            df["time"] = (datetime - datetime.min()).dt.total_seconds()

            # Split data into train and test
            train_df = df[datetime < config["train_cutoff_date"]]
            test_df = df[datetime >= config["train_cutoff_date"]]

            # Append to list
            train_seqs.append(train_df.values)
            test_seqs.append(test_df.values)

    # Create a training dataset and dataloader
    print("Creating training dataset and dataloader...")
    train_set = TimeSeriesDataset(
        train_seqs,
        config["segment_length"],
        step=config["segment_step"],
        device=device,
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    # Create a testing dataset and dataloader
    print("Creating testing dataset and dataloader...")
    test_set = TimeSeriesDataset(
        test_seqs,
        config["segment_length"],
        step=config["segment_step"],
        device=device,
    )
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True)

    total_samples = len(train_set) + len(test_set)
    percent_train = 100 * len(train_set) / total_samples
    percent_test = 100 * len(test_set) / total_samples
    print(
        f"{len(train_set)} training samples and {len(test_set)} testing samples created ({percent_train:.1f}/{percent_test:.1f} split).\n"
    )

    # Initialize models and optimizers
    input_size = train_seqs[0].shape[-1] - 1  # subtract 1 for the label column

    print("Initializing Encoder Model...")
    encoder = LstmEncoder(
        input_size=input_size,
        hidden_size=config["hidden_size"],
    ).to(device)

    print("Summary:")
    summary(
        encoder,
        input_data=(
            config["segment_length"],
            input_size,
        ),
    )
    print()

    print("Initializing Projector Model...")
    projector = DenseLayers(
        input_size=config["hidden_size"],
        output_size=config["proj_size"],
        intermediates=config["proj_layers"],
    ).to(device)

    model_params = list(encoder.parameters()) + list(projector.parameters())
    simclr_optimizer = Adam(model_params, lr=config["simclr_lr"])

    print("Summary:")
    summary(
        projector,
        input_data=(config["representation_size"],),
    )
    print()

    print("Initializing linear probe...")
    probe = DenseLayers(
        input_size=config["representation_size"],
        output_size=config["probe_size"],
        intermediates=config["probe_layers"],
    ).to(device)

    probe_optimizer = Adam(probe.parameters(), lr=config["probe_lr"])

    print("Summary:")
    summary(
        probe,
        input_data=(config["representation_size"],),
    )
    print()

    print("Initializing baseline model...")
    baseline_model = nn.Sequential(deepcopy(encoder), deepcopy(probe)).to(device)
    baseline_optimizer = Adam(baseline_model.parameters(), lr=config["baseline_lr"])

    # Training loop
    print("Starting training loop...")
    n_epochs = config["n_epochs"]
    training_run(
        encoder,
        projector,
        probe,
        baseline_model,
        train_loader,
        test_loader,
        simclr_optimizer,
        probe_optimizer,
        baseline_optimizer,
        n_epochs,
    )

    wandb.finish()
