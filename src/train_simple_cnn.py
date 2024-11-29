import argparse
import json
import logging
import os
import pprint
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import wandb
from util.datasets import TimeSeriesDataset
from util.models import DenseLayers, LstmEncoder, CnnEncoder
from util.training import finetuning_epoch, simclr_epoch, training_epoch

from train_simple_lstm import get_args, recursive_zip, load_config, configure_logger, build_dataloader, create_dataloaders, initialize_probe


############################# GLOBAL VARIABLES #####################################################
load_dotenv()

# defaults
DEFAULT_CONFIG = "configs/simple-cnn.yaml"
DEFAULT_CHECKPOINTS_DIR = (
    f"checkpoints/simple-cnn/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
)

# WandB API key
WANDB_KEY = os.getenv("WANDB_API_KEY")
assert WANDB_KEY is not None, "WANDB_API_KEY environment variable not found."
os.environ["WANDB_API_KEY"] = WANDB_KEY

# Logger
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/simple-cnn_{timestamp}.log"

# Device on which to load data and parameters (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# FUNCTIONS ############################################################
def get_args():
    """
    Parse command line arguments.

    Returns:
    - args: the parsed arguments
    """
    arg_parser = argparse.ArgumentParser(
        description="Train and evaluate two models."
        "One model is trained with SimCLR and finetuned on a downstream supervised task and the other is trained with a supervised loss."
        "The models are both evaluated on a downstream task."
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to experiment config file",
    )
    arg_parser.add_argument(
        "--config_override",
        type=str,
        default=None,
        help="JSON string of parameters to override config file",
    )
    arg_parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to training data",
    )
    arg_parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to testing data",
    )
    arg_parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Path to save model checkpoints",
    )
    arg_parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment",
    )
    arg_parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags for the experiment",
    )
    arg_parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train",
    )
    arg_parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The log level to use",
    )
    arg_parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="The log file to write to",
    )

    return arg_parser.parse_args()


def initialize_encoder(config):
    """
    Build the encoder model from the configuration.

    Args:
    - config: the configuration dictionary for the encoder

    Returns:
    - encoder: the encoder model
    """
    return CnnEncoder(
        input_size=config["input_size"],
        out_channels=config["out_channels"],
        kernel_size=config["kernel_size"],
    ).to(device)

def initialize_probe(config):
    """
    Build the probe model from the configuration.

    Args:
    - config: the configuration dictionary for the probe

    Returns:
    - probe: the probe model
    """
    return DenseLayers(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_layers=config["hidden_layers"],
    ).to(device)


def initialize_projector(config):
    """
    Build the projector model from the configuration.

    Args:
    - config: the configuration dictionary for the projector

    Returns:
    - projector: the projector model
    """
    return DenseLayers(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hidden_layers=config["hidden_layers"],
    ).to(device)


def initialize_models(config):
    """
    Initialize the encoder, projector, probe, and base model.

    Args:
    - config: the configuration dictionary for the models

    Returns:
    - encoder: the encoder model
    - projector: the projector model
    - probe: the probe model
    - base_model: the base model
    """
    print("Initializing Encoder Model...")
    encoder = initialize_encoder(config["encoder"])
    print("Initializing Projector Model...")
    projector = initialize_projector(config["projector"])
    print("Initializing Probe Model...")
    probe = initialize_probe(config["probe"])
    print("Initializing Base Model...")
    base_model = nn.Sequential(deepcopy(encoder), deepcopy(probe)).to(device)
    return encoder, projector, probe, base_model


def model_summaries(models, input_shape):
    """
    Print summaries of the models.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - input_shape: the shape of the input data
    """

    encoder, projector, _, base_model = models  # probe is not used here

    print("\nSimCLR Model Summmary:")
    simclr_model = nn.Sequential(deepcopy(encoder), deepcopy(projector))
    desc = summary(simclr_model, input_data=input_shape)
    print()
    logger.info(f"SimCLR Model Summary:\n{desc}")

    print("Downstream Regression Model Summary:")
    desc = summary(base_model, input_data=input_shape)
    print()
    logger.info(f"Downstream Regression Model Summary:\n{desc}")


def initialize_optimizers(models, config):
    """
    Initialize the optimizers for the encoder, projector, probe, and base model.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - config: the configuration dictionary for training

    Returns:
    - simclr_optimizer: the optimizer for the encoder and projector
    - probe_optimizer: the optimizer for the probe
    - base_optimizer: the optimizer for the base model
    """
    encoder, projector, probe, base_model = models
    print("Initializing Encoder/Projector Optimizer...")
    model_params = list(encoder.parameters()) + list(projector.parameters())
    simclr_optimizer = Adam(model_params, lr=config["simclr_lr"])
    print("Initializing Probe Optimizer...")
    probe_optimizer = Adam(probe.parameters(), lr=config["finetuning_lr"])
    print("Initializing Base Model Optimizer...")
    base_optimizer = Adam(base_model.parameters(), lr=config["baseline_lr"])
    print()
    return simclr_optimizer, probe_optimizer, base_optimizer


def train_models(models, optimizers, train_loader, config, epoch):
    """
    Run a single training epoch for all models.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - optimizers: a tuple of the optimizers for the encoder, projector, probe, and base model
    - train_loader: the training DataLoader
    - config: the configuration dictionary
    - epoch: the current epoch number (for logging)

    Returns:
    - simclr_training_loss: the training loss for the encoder and projector
    - finetuning_training_loss: the training loss for the probe
    - baseline_training_loss: the training loss for the base model
    """

    encoder, projector, probe, base_model = models
    simclr_optimizer, probe_optimizer, base_optimizer = optimizers

    print("Training Encoder via SimCLR...")
    batch_callback = lambda i, loss: wandb.log(
        {"SimCLR Batch Loss": loss, "Batch": epoch * len(train_loader) + i}
    )
    simclr_training_loss = simclr_epoch(
        encoder,
        projector,
        train_loader,
        simclr_optimizer,
        config["temperature"],
        batch_callback=batch_callback,
    )
    logger.info(f"Epoch {epoch} --- SimCLR Training Loss: {simclr_training_loss:.4e}")

    print("Finetuning with Linear Probe...")
    batch_callback = lambda i, loss: wandb.log(
        {"Finetuning Batch Loss": loss, "Batch": epoch * len(train_loader) + i}
    )
    finetuning_training_loss = finetuning_epoch(
        encoder,
        probe,
        train_loader,
        nn.MSELoss(),
        probe_optimizer,
        batch_callback=batch_callback,
    )
    logger.info(
        f"Epoch {epoch} --- Finetuning Training Loss: {finetuning_training_loss:.4e}"
    )

    print("Training Supervised Baseline Model...")
    batch_callback = lambda i, loss: wandb.log(
        {"Baseline Batch Loss": loss, "Batch": epoch * len(train_loader) + i}
    )
    baseline_training_loss = training_epoch(
        base_model,
        train_loader,
        nn.MSELoss(),
        base_optimizer,
        batch_callback=batch_callback,
    )
    logger.info(
        f"Epoch {epoch} --- Baseline Training Loss: {baseline_training_loss:.4e}"
    )

    return simclr_training_loss, finetuning_training_loss, baseline_training_loss


def test_models(models, test_loader, epoch):
    """
    Test the finetuned model and the baseline model on the test data.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - test_loader: the testing DataLoader
    - epoch: the current epoch number (for logging)

    Returns:
    - finetuning_test_loss: the validation loss for the finetuned model
    - baseline_test_loss: the validation loss for the baseline model
    """

    encoder, _, probe, base_model = models

    print("Testing Finetuned model...")
    finetuned_test_loss = 0
    encoder.eval()
    probe.eval()
    pbar = tqdm(test_loader)
    for x, y in pbar:
        z = encoder(x)
        y_pred = probe(z)
        y_true = y[:, -1]
        finetuned_test_loss += nn.MSELoss()(y_pred, y_true).item()
    finetuned_test_loss /= len(test_loader)
    logger.info(
        f"Epoch {epoch} --- Finetuned Model Test Loss: {finetuned_test_loss:.4e}"
    )
    print(f"Finetuned Model Test Loss: {finetuned_test_loss:.4e}")

    print("Testing Baseline Model...")
    baseline_test_loss = 0
    base_model.eval()
    pbar = tqdm(test_loader)
    for x, y in pbar:
        y_pred = base_model(x)
        y_true = y[:, -1]
        baseline_test_loss += nn.MSELoss()(y_pred, y_true).item()
    baseline_test_loss /= len(test_loader)
    logger.info(f"Epoch {epoch} --- Baseline Model Test Loss: {baseline_test_loss:.4e}")
    print(f"Baseline Model Test Loss: {baseline_test_loss:.4e}")

    return finetuned_test_loss, baseline_test_loss


def save_model_checkpoints(models, config, epoch):
    """
    Save model checkpoints for the encoder, projector, probe, and base model.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - config: the configuration dictionary
    - epoch: the current epoch number
    """

    encoder, projector, probe, base_model = models

    print("Saving model checkpoints...")
    epoch_checkpoint = f"{config['checkpoints_dir']}/epoch_{epoch+1:03}"
    if not os.path.exists(epoch_checkpoint):
        os.makedirs(epoch_checkpoint)

    torch.save(encoder.state_dict(), f"{epoch_checkpoint}/encoder.pth")
    torch.save(projector.state_dict(), f"{epoch_checkpoint}/projector.pth")
    torch.save(probe.state_dict(), f"{epoch_checkpoint}/probe.pth")
    torch.save(base_model.state_dict(), f"{epoch_checkpoint}/base_model.pth")
    print(f"Model checkpoints saved to {epoch_checkpoint}")
    logger.info(f"Model checkpoints saved to {epoch_checkpoint}")


def experiment_run(models, optimizers, train_loader, test_loader, config):
    print("Starting Experiment Run...")
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_losses = train_models(models, optimizers, train_loader, config, epoch)
        simclr_train_loss, finetuning_train_loss, base_train_loss = train_losses

        test_losses = test_models(models, test_loader, epoch)
        finetuning_test_loss, base_test_loss = test_losses

        wandb.log(
            {
                "Epoch": epoch,
                "SimCLR Training Loss": simclr_train_loss,
                "Finetuned Training Loss": finetuning_train_loss,
                "Baseline Training Loss": base_train_loss,
                "Finetuned Test Loss": finetuning_test_loss,
                "Baseline Test Loss": base_test_loss,
            }
        )

        save_model_checkpoints(models, config, epoch)

    wandb.finish()

############################# MAIN FUNCTION #######################################################


def main(config):
    """
    Run the main training loop.

    Args:
    - config: the configuration dictionary
    """
    logger.info(f"Configuration:\n{pprint.pformat(config)}")

    # Start experiment on WandB
    wandb.init(project="simcl-stock-embedding", config=config, **config["experiment"])

    # Load data
    train_loader, test_loader = create_dataloaders(config["data"])

    # Set the input size of the encoder based on the data
    input_size = next(iter(train_loader))[0].shape[-1]
    config["models"]["encoder"]["input_size"] = input_size

    # Initialize models
    models = initialize_models(config["models"])
    print(config["models"])

    # Model summaries
    segment_length = config["data"]["segment_length"]
    input_shape = (segment_length, input_size)
    model_summaries(models, input_shape)

    # Initialize optimizers
    optimizers = initialize_optimizers(models, config["optimizers"])

    # Start the experiment
    experiment_run(models, optimizers, train_loader, test_loader, config["training"])


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()

    # Configure the logger
    configure_logger(args.log_level, args.log_file)

    # Log the arguments
    formatted_args = "\n\t".join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments:\n\t{formatted_args}\n")

    # Run the main function
    config = load_config(args)
    main(config)