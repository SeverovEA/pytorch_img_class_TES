"""
Contains methods that create various objects. E.g.: DataLoaders, Pytorch model objects,
"""
import os
from datetime import datetime
from typing import Tuple, List

import torch
import torch.utils.data
import torchvision.models
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

NUM_WORKERS = 0  # set as os.cpu_count() for multiprocessing or 0 for single process


def create_datasets_from_classes_in_folders(
        data_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List]:
    """
    :param data_dir: Path to data directory
    :param transform: Transforms that will be applied to images
    :param batch_size: Number of images in each batch
    :param num_workers: Number of subprocesses that will handle data loading
    :return: A tuple of train dataloader, test dataloader, and a list of class names
    """
    full_data = datasets.ImageFolder(data_dir, transform=transform)
    train_data, test_data = torch.utils.data.random_split(full_data, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    class_names = full_data.classes
    return train_dataloader, test_dataloader, class_names


def create_efficientnet_model(
        model_version: str,
        device: str,
) -> Tuple[any, torchvision.models.efficientnet.EfficientNet, torchvision.transforms._presets.ImageClassification]:
    """
    Takes effnet model affix (b0, b1, etc. or v2_s, v2_m, v2_l) and device ("cpu" or "cuda")
    and returns a tuple of weights, a model object, and transforms from torchvision models library.
    :param model_version: a string with an affix of effnet model name (e.g. "b0" or "v2_l")
    :param device: a string specifying pytroch device ("cpu" or "cuda")
    :return: a tuple of weights, a model object, and transforms
    """
    weights_class = getattr(torchvision.models, f"EfficientNet_{model_version.upper()}_Weights")
    weights = getattr(weights_class, "IMAGENET1K_V1")
    model_class = getattr(torchvision.models, f"efficientnet_{model_version.lower()}")
    model = model_class(weights=weights).to(device)
    preprocess = weights.transforms(
        antialias=True)  # antialias=True passed to supress a warning (will be deprecated in torchvision 0.17+)

    return weights, model, preprocess


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name += datetime.now().strftime("_%H-%M-%S")

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
