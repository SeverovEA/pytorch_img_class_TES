"""
Contains various utility functions.
"""
import warnings
from pathlib import Path
from typing import Tuple, List

import torch
import yaml


def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str
):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A path to directory for saving the model to.
    model_name: A filename for the saved model. Must be ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="model_name.pth")
    """
    # Create target directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Check that extension is correct and create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def set_seeds(seed: int = 42) -> None:
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def set_device() -> str:
    """
    Returns "cuda" by default. If cuda is not available returns "cpu" and raises a warning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA is not available, something's wrong :-(")
    return device


def change_classifier(
        in_features: int,
        out_features: int,
        device: str,
        dropout_rate: float = 0.2
) -> torch.nn.Sequential:
    """
    Changes number of classifier output features.
    Optionally changes dropout rate.

    :return: Changed classifier.
    """
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_rate, inplace=True),
        torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        )).to(device)
    return classifier


def yamls_to_dict(filepaths: Tuple[str, ...]) -> List[dict]:
    """
    Takes tuple of paths .yaml files, returns list of according dictionaries.
    """
    dicts_list = []
    for path in filepaths:
        with open(path, "r") as stream:
            dicts_list.append(yaml.safe_load(stream))
    return dicts_list
