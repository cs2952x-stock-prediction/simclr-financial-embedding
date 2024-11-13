import argparse
import os
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
from dotenv import load_dotenv
from pandas.io.parquet import json
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

############################# GLOBAL VARIABLES #####################################################

DEFAULT_CONFIG = {
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
    "simclr_lr": 1e-6,  # learning rate for encoder and projector during SimCLR training
    "probe_lr": 1e-3,  # learning rate for linear probe appended to encoder trained with SimCLR
    "baseline_lr": 1e-3,  # learning rate for baseline supervised model
    "batch_size": 128,
    "n_epochs": 200,
    "temperature": 0.5,
    # data source+structure config
    "train_dir": "data/processed/kaggle_sp500/out/train",
    "test_dir": "data/processed/kaggle_sp500/out/test",
    "training_cutoff": "2023-01-01",
    "segment_length": 30,
    "segment_step": 5,
    "checkpoints_dir": "checkpoints",
}


load_dotenv()
WANDB_KEY = os.getenv("WANDB_API_KEY")
assert WANDB_KEY is not None, "WANDB_API_KEY environment variable not found"
os.environ["WANDB_API_KEY"] = WANDB_KEY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# FUNCTIONS ############################################################


def load_data(
    train_dir, test_dir, batch_size, segment_length, segment_step, device, y_col="close"
):
    train_sequences, train_labels = [], []
    for filename in tqdm(os.listdir(train_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(train_dir, filename))

            # Define x and y columns
            x_cols = [col for col in df.columns if col != y_col]
            y_cols = [y_col]

            # Append to lists
            train_sequences.append(df[x_cols].values)
            train_labels.append(df[y_cols].values)

    test_sequences, test_labels = [], []
    for filename in tqdm(os.listdir(test_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(test_dir, filename))

            # Define x and y columns
            x_cols = [col for col in df.columns if col != y_col]
            y_cols = [y_col]

            # Append to lists
            test_sequences.append(df[x_cols].values)
            test_labels.append(df[y_cols].values)

    # Create a training dataset and dataloader
    train_set = TimeSeriesDataset(
        train_sequences, train_labels, segment_length, segment_step, device
    )
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    # Create a testing dataset and dataloader
    test_set = TimeSeriesDataset(
        test_sequences, test_labels, segment_length, segment_step, device
    )
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    return train_loader, test_loader


def freeze(model):
    """
    Freezes all parameters of a model.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    """
    Unfreezes all parameters of a model.
    """
    for param in model.parameters():
        param.requires_grad = True


def simclr_training_epoch(
    encoder, projector, dataloader, optimizer, temp, start_batch=0
):
    """
    Performs a single epoch of training on the SimCLR model.
    This model is trained with a contrastive loss function that encourages similar representations for augmented views of the same data.
    Both the enocder and projector are trained in this step (loss backpropogates through both).

    Args:
    - encoder: the encoder model to train
    - projector: the projector model to train
    - dataloader: the dataloader to use for training
    - optimizer: the optimizer to use
    - temp: the temperature parameter for the contrastive loss
    - start_batch: the index of the first batch in the epoch (relative to all previous training runs --- used for logging)

    Returns:
    - the average loss across the epoch
    """
    encoder.train()
    projector.train()

    pbar = tqdm(dataloader)
    total_loss = 0
    for i, (x, _) in enumerate(pbar):
        # transform data to create two views
        x_i = mask_with_added_gaussian(x, mask_prob=1.0, std_multiplier=0.1)
        x_j = x

        # create representations
        h_i = encoder(x_i)
        h_j = encoder(x_j)

        # project representations
        z_i = projector(h_i)
        z_j = projector(h_j)

        # compute loss
        loss = nt_xent_loss(z_i, z_j, temp)
        total_loss += loss.item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # the last batch may not be full
        # which can throw off the average nt-xent loss value
        if i + 1 < len(dataloader):
            pbar.set_postfix(
                {
                    "Batch Loss": f"{loss.item():.4e}",
                    "Epoch Loss": f"{total_loss / (i+1):.4e}",
                }
            )

            wandb.log(
                {
                    "SimCLR Batch Loss": loss.item(),
                    "SimCLR Batch Index": start_batch + i,
                }
            )

    return total_loss / len(dataloader)


def finetuning_training_epoch(
    encoder, probe, dataloader, criterion, optimizer, start_batch=0
):
    """
    Performs a single epoch of training on the finetuning model.
    This model is trained with a supervised loss function that encourages the model to predict the next value in the sequence.
    Only the probe is trained in this step (encoder is frozed and loss backpropogates through the probe only).

    Args:
    - encoder: the encoder model to use
    - probe: the probe model to train
    - dataloader: the dataloader to use for training
    - criterion: the loss function to use
    - optimizer: the optimizer to use for the probe
    - start_batch: the index of the first batch in the epoch (relative to all previous training runs --- used for logging)

    Returns:
    - the average loss across the epoch
    """

    encoder.eval()
    probe.train()

    freeze(encoder)

    pbar = tqdm(dataloader)
    total_loss = 0
    for i, (x, y) in enumerate(pbar):
        y_pred = probe(encoder(x))
        y_true = y[:, -1]  # the last values in the sequences
        loss = criterion(y_pred, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item():.4e}",
                "Epoch Loss": f"{total_loss / (i+1):.4e}",
            }
        )

        wandb.log(
            {
                "Finetuning Batch Loss": loss.item(),
                "Finetuning Batch Index": start_batch + i,
            }
        )

    unfreeze(encoder)

    return total_loss / len(dataloader)


def baseline_training_epoch(model, dataloader, criterion, optimizer, start_batch=0):
    """
    Performs a single epoch of training on the baseline model.
    This model is trained directly on the downstream task and the loss is allowed to backpropogate fully through the model.

    Args:
    - model: the model to train
    - dataloader: the dataloader to use for training
    - criterion: the loss function to use
    - optimizer: the optimizer to use
    - start_batch: the index of the first batch in the epoch (relative to all previous training runs --- used for logging)

    Returns:
    - the average loss across the epoch
    """
    model.train()

    pbar = tqdm(dataloader)
    total_loss = 0
    for i, (x, y) in enumerate(pbar):
        y_pred = model(x)
        y_true = y[:, -1]  # the last values in the sequences
        loss = criterion(y_pred, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #
        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item():.4e}",
                "Epoch Loss": f"{total_loss / (i+1):.4e}",
            }
        )

        wandb.log(
            {
                "Baseline Batch Loss": loss.item(),
                "Baseline Batch Index": start_batch + i,
            }
        )

    return total_loss / len(dataloader)


############################# EXECUTION ############################################################

if __name__ == "__main__":
    config = DEFAULT_CONFIG
    device = DEVICE

    wandb.init(
        project="simcl-stock-embedding",
        name="simple-lstm-encoder",
        config=DEFAULT_CONFIG,
    )

    # Load data
    train_dir = config["train_dir"]
    test_dir = config["test_dir"]
    training_cutoff = config["training_cutoff"]

    print(f"Loading training data from {train_dir}")
    print(f"Loading testing data from {test_dir}")
    print(f"All data prior to {training_cutoff} will be used for training.")

    batch_size, segment_length, segment_step = (
        config["batch_size"],
        config["segment_length"],
        config["segment_step"],
    )
    train_loader, test_loader = load_data(
        config["train_dir"],
        config["test_dir"],
        batch_size,
        segment_length,
        segment_step,
        device,
    )

    train_sz, test_sz = len(train_loader), len(test_loader)
    total_samples = train_sz + test_sz
    percent_train = 100 * train_sz / total_samples
    percent_test = 100 * test_sz / total_samples
    print(
        f"{train_sz} training samples and {test_sz} testing samples ({percent_train:.1f}/{percent_test:.1f} split).\n"
    )

    # Initialize models and optimizers
    x, y = next(iter(train_loader))
    input_size = x.shape[-1]

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

    print("Initializing Linear Probe...")
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

    print("Initializing Baseline Model...")
    baseline_model = nn.Sequential(deepcopy(encoder), deepcopy(probe)).to(device)
    baseline_optimizer = Adam(baseline_model.parameters(), lr=config["baseline_lr"])

    checkpoints_dir = config["checkpoints_dir"]
    timenow = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoints_dir, str(timenow))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # save config as json
    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(config, f)

    # Training loop
    print("Starting training loop...")
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("Training Encoder via SimCLR...")
        simclr_training_loss = simclr_training_epoch(
            encoder,
            projector,
            train_loader,
            simclr_optimizer,
            config["temperature"],
            epoch * train_sz,
        )

        print("Finetuning with Linear Probe...")
        finetuning_training_loss = finetuning_training_epoch(
            encoder,
            probe,
            train_loader,
            nn.MSELoss(),
            probe_optimizer,
            epoch * train_sz,
        )

        print("Training Supervised Baseline Model...")
        baseline_training_loss = baseline_training_epoch(
            baseline_model,
            train_loader,
            nn.MSELoss(),
            baseline_optimizer,
            epoch * train_sz,
        )

        print("Evaluting finetuned model...")
        finetuning_val_loss = 0
        encoder.eval()
        probe.eval()
        pbar = tqdm(test_loader)
        for x, y in pbar:
            y_pred = probe(encoder(x))
            y_true = y[:, -1]
            finetuning_val_loss += nn.MSELoss()(y_pred, y_true).item()
        finetuning_val_loss /= len(test_loader)

        print("Evaluating Baseline Model...")
        baseline_val_loss = 0
        baseline_model.eval()
        pbar = tqdm(test_loader)
        for x, y in pbar:
            y_pred = baseline_model(x)
            y_true = y[:, -1]
            baseline_val_loss += nn.MSELoss()(y_pred, y_true).item()
        baseline_val_loss /= len(test_loader)

        print(f"Finetuned Model Validation Loss: {finetuning_val_loss:.4e}")
        print(f"Supervised Baseline Validation Loss: {baseline_val_loss:.4e}")

        wandb.log(
            {
                "Epoch": epoch + 1,
                "SimCLR Training Loss": simclr_training_loss,
                "Finetuning Training Loss": finetuning_training_loss,
                "Baseline Training Loss": baseline_training_loss,
                "Finetuning Validation Loss": finetuning_val_loss,
                "Baseline Validation Loss": baseline_val_loss,
            }
        )

        print("Saving model checkpoints...")
        epoch_checkpoint_path = os.path.join(checkpoint_path, f"epoch_{epoch+1:03}")
        if not os.path.exists(epoch_checkpoint_path):
            os.makedirs(epoch_checkpoint_path)

        torch.save(
            encoder.state_dict(),
            os.path.join(epoch_checkpoint_path, f"encoder.pth"),
        )
        torch.save(
            projector.state_dict(),
            os.path.join(epoch_checkpoint_path, f"projector.pth"),
        )
        torch.save(
            probe.state_dict(),
            os.path.join(epoch_checkpoint_path, f"probe.pth"),
        )
        torch.save(
            baseline_model.state_dict(),
            os.path.join(epoch_checkpoint_path, f"baseline.pth"),
        )

    wandb.finish()
