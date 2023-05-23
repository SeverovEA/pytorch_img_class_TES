from typing import Tuple, List
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0  # set as os.cpu_count() for multiprocessing or 0 for single process


def create_datasets_from_classes_in_folders(
        data_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS

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
