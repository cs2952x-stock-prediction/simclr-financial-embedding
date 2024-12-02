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
from pandas.core.generic import pickle
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

import wandb
from util.datasets import TimeSeriesDataset
from util.evaluation import average_percentage_error
from util.models import DenseLayers, LstmEncoder, CnnEncoder
from util.training import finetuning_epoch, simclr_epoch, training_epoch
from train_simple_lstm import (
    get_args, recursive_zip, load_config, configure_logger, build_dataloader, create_dataloaders, 
    initialize_probe, initialize_projector, initialize_models, model_summaries, initialize_optimizers,
    train_models, test_models, save_model_checkpoints, experiment_run,
)




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
        "--scaler_file",
        type=str,
        help="Path to data scaler pickle file",
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
    arg_parser.add_argument(
        "-x",
        "--features",
        type=str,
        nargs="+",
        default=["open", "close", "high", "low", "volume"],
        help="The features to use for training",
    )
    arg_parser.add_argument(
        "-y",
        "--targets",
        type=str,
        nargs="+",
        default=["close"],
        help="The target variables to predict",
    )

    return arg_parser.parse_args()

def load_config(args):
    """
    Load a configuration file.

    Args:
    - args: the parsed arguments

    Returns:
    - config: the configuration dictionary
    """

    # The 'normal' config file contains the specific configuration for this run
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args.config = None

    # Override configuration with command line arguments
    if args.train_dir is not None:
        config["data"]["train_dir"] = args.train_dir

    if args.test_dir is not None:
        config["data"]["test_dir"] = args.test_dir

    if args.scaler_file is not None:
        config["data"]["scaler_file"] = args.scaler_file

    if args.checkpoints_dir is not None:
        config["training"]["checkpoints_dir"] = args.checkpoints_dir

    if args.name is not None:
        config["experiment"]["name"] = args.name

    # tags are appended rather than replaced
    if args.tags is not None:
        config["experiment"]["tags"].extend(args.tags)

    if args.num_epochs is not None:
        config["training"]["n_epochs"] = args.num_epochs

    if args.features is not None:
        config["data"]["features"] = args.features

    if args.targets is not None:
        config["data"]["targets"] = args.targets

    # Override configuration with JSON string of parameters
    if args.config_override is not None:
        override = json.loads(args.config_override)
        config = recursive_zip(config, override)

    logger.info(f"Configuration:\n{pprint.pformat(config)}")

    return config


def initialize_encoder(config):
    """
    Build the encoder model from the configuration.

    Args:
    - config: the configuration dictionary for the encoder

    Returns:
    - encoder: the encoder model
    """
    return CnnEncoder(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        kernel_size=config["kernel_size"],
        num_layers = config["num_layers"],
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


def test_models(models, test_loader, epoch, scaler: StandardScaler):
    """
    Test the finetuned model and the baseline model on the test data.

    Args:
    - models: a tuple of the encoder, projector, probe, and base model
    - test_loader: the testing DataLoader
    - epoch: the current epoch number (for logging)
    - scaler: a StandardScaler object for the target variable

    Returns:
    - finetuning_test_loss: the validation loss for the finetuned model
    - baseline_test_loss: the validation loss for the baseline model
    """

    encoder, _, probe, base_model = models

    print("Testing Finetuned model...")
    finetuned_test_loss = 0

    # TODO: this is a quick patch that should be made more general later
    # We should not assume that the target variable is the last column
    # Extracting the scale and mean of the target variable is also bad form
    scale = scaler.scale_[-1]  # type: ignore
    mean = scaler.mean_[-1]  # type: ignore

    finetuned_perc_error = 0
    encoder.eval()
    probe.eval()
    pbar = tqdm(test_loader)
    for x, y in pbar:
        z = encoder(x)
        y_pred = probe(z)
        y_true = y[:, -1]
        finetuned_test_loss += nn.MSELoss()(y_pred, y_true).item()
        finetuned_perc_error += average_percentage_error(
            y_true * scale + mean, y_pred * scale + mean
        ).item()
    finetuned_test_loss /= len(test_loader)
    finetuned_perc_error /= len(test_loader)
    logger.info(
        f"Epoch {epoch} --- Finetuned Model Test Loss: {finetuned_test_loss:.4e}"
    )
    print(f"Finetuned Model Test Loss: {finetuned_test_loss:.4e}")

    print("Testing Baseline Model...")
    baseline_perc_error = 0
    baseline_test_loss = 0
    base_model.eval()
    pbar = tqdm(test_loader)
    for x, y in pbar:
        y_pred = base_model(x)
        y_true = y[:, -1]
        baseline_test_loss += nn.MSELoss()(y_pred, y_true).item()
        baseline_perc_error += average_percentage_error(
            y_true * scale + mean, y_pred * scale + mean
        ).item()
    baseline_test_loss /= len(test_loader)
    baseline_perc_error /= len(test_loader)
    logger.info(f"Epoch {epoch} --- Baseline Model Test Loss: {baseline_test_loss:.4e}")
    print(f"Baseline Model Test Loss: {baseline_test_loss:.4e}")

    return (
        finetuned_test_loss,
        finetuned_perc_error,
        baseline_test_loss,
        baseline_perc_error,
    )


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


def experiment_run(models, optimizers, train_loader, test_loader, config, scaler):
    print("Starting Experiment Run...")
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_losses = train_models(models, optimizers, train_loader, config, epoch)
        simclr_train_loss, finetuning_train_loss, baseline_train_loss = train_losses

        test_losses = test_models(models, test_loader, epoch, scaler)
        (
            finetuning_test_loss,
            finetined_perc_error,
            baseline_test_loss,
            baseline_perc_error,
        ) = test_losses

        wandb.log(
            {
                "Epoch": epoch,
                "SimCLR Training Loss": simclr_train_loss,
                "Finetuned Training Loss": finetuning_train_loss,
                "Baseline Training Loss": baseline_train_loss,
                "Finetuned Test Loss": finetuning_test_loss,
                "Baseline Test Loss": baseline_test_loss,
                "Finetuned Test APE": finetined_perc_error,
                "Baseline Test APE": baseline_perc_error,
            }
        )

        save_model_checkpoints(models, config, epoch)

    wandb.finish()

def get_final_conv_output_size(input_length, kernel_size, num_layers, stride = 1):
    out_len = input_length
    for _ in range(num_layers):
        out_len = (out_len - kernel_size) // stride + 1
    return out_len

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

    # Set the input channels of the CNN encoder based on the data
    input_size = next(iter(train_loader))[0].shape[-1]
    config["models"]["encoder"]["in_channels"] = len(config["data"]["features"])

    seq_len = config["data"]["segment_length"]
    encoder_kernel_size = config["models"]["encoder"]["kernel_size"]
    encoder_num_layers = config["models"]["encoder"]["num_layers"]

    encoder_output_size = get_final_conv_output_size(seq_len, encoder_kernel_size, encoder_num_layers, stride = 1 )
    config["models"]["encoder"]["output_size"] = encoder_output_size

    embedding_size = encoder_output_size * config["models"]["encoder"]["out_channels"]
    config["models"]["projector"]["input_size"] = embedding_size
    config["models"]["probe"]["input_size"] = embedding_size

    

    # Initialize models
    models = initialize_models(config["models"])

    # Model summaries
    segment_length = config["data"]["segment_length"]
    input_shape = (segment_length, input_size)
    model_summaries(models, input_shape)

    # Initialize optimizers
    optimizers = initialize_optimizers(models, config["optimizers"])

    # Start the experiment
    scaler = pickle.load(open(config["data"]["scaler_file"], "rb"))
    experiment_run(
        models, optimizers, train_loader, test_loader, config["training"], scaler
    )


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
