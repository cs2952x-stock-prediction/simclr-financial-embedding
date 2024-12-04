import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import TimeSeriesDataset
from .evaluation import nt_xent_loss
from .transformations import mask_with_added_gaussian
from .transformations import mask_with_added_smoothing
############################# GLOBAL VARIABLES #####################################################

# Logger
logger = logging.getLogger(__name__)

# Device on which to load data and parameters (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# FUNCTIONS ############################################################


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


def build_dataloader(data_dir, batch_size, segment_length, segment_step, y_col="close"):
    """
    Loads data from a directory of CSV files and creates a DataLoader for training and testing.

    Args:
    - data_dir: the directory from which to load the data
    - batch_size: the batch size for the DataLoader
    - segment_length: the length of each segment of data
    - segment_step: the step size for each segment
    - y_col: the column to use as the target variable

    Returns:
    - a DataLoader of the data
    """
    sequences, labels = [], []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, filename))

            # Define x and y columns
            x_cols = [col for col in df.columns if col != y_col]
            y_cols = [y_col]

            # Append to lists
            sequences.append(df[x_cols].values)
            labels.append(df[y_cols].values)

    # Create a testing dataset and dataloader
    dataset = TimeSeriesDataset(sequences, labels, segment_length, segment_step)
    return DataLoader(dataset, batch_size, shuffle=True)


def simclr_epoch(encoder, projector, dataloader, optimizer, temp, **kwargs):
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

    Returns:
    - the average loss across the epoch
    """
    encoder.train()
    projector.train()

    pbar = tqdm(dataloader)
    total_loss = 0
    for i, (x, _) in enumerate(pbar):
        # transform data to create two views
        #x_i = mask_with_added_gaussian(x, mask_prob=1.0, std_multiplier=0.1)
        x_i = mask_with_added_smoothing(x, mask_prob=1.0, smoothing_factor=0.1)
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

            if "batch_callback" in kwargs:
                kwargs["batch_callback"](i, loss.item())

    return total_loss / len(dataloader)


def finetuning_epoch(encoder, probe, dataloader, criterion, optimizer, **kwargs):
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

        if "batch_callback" in kwargs:
            kwargs["batch_callback"](i, loss.item())

    unfreeze(encoder)

    return total_loss / len(dataloader)


def training_epoch(model, dataloader, loss_fn, optimizer, **kwargs):
    """
    Performs a single epoch of training on the baseline model.
    This model is trained directly on the downstream task and the loss is allowed to backpropogate fully through the model.

    Args:
    - model: the model to train
    - dataloader: the dataloader to use for training
    - loss_fn: the loss function to use
    - optimizer: the optimizer to use

    Returns:
    - the average loss across the epoch
    """
    model.train()

    pbar = tqdm(dataloader)
    total_loss = 0
    for i, (x, y) in enumerate(pbar):
        y_pred = model(x)
        y_true = y[:, -1]  # the last values in the sequences
        loss = loss_fn(y_pred, y_true)
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

        if "batch_callback" in kwargs:
            kwargs["batch_callback"](i, loss.item())

    return total_loss / len(dataloader)
