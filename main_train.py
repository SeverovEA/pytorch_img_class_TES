import utils
import torch
import torchvision
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.io import read_image
import torchinfo
from torchinfo import summary
import data_setup
import engine
from pathlib import Path
from torch import nn
from timeit import default_timer as timer
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import warnings
import multiprocessing
import create_effnet
import configparser
import yaml

# Open config files and set variables
CONFIG_FILENAME = "config.yaml"
SETTINGS_FILENAME = "settings.yaml"
with open(SETTINGS_FILENAME, "r") as stream:
    settings_dict = yaml.safe_load(stream)
with open(CONFIG_FILENAME, "r") as stream:
    config_dict = yaml.safe_load(stream)
NUM_EPOCHS = int(config_dict["Hyperparameters"]["NUM_EPOCHS"])
LEARNING_RATE = int(config_dict["Hyperparameters"]["LEARNING_RATE"])
BATCH_SIZE = int(config_dict["Hyperparameters"]["BATCH_SIZE"])
EXP_NAME = config_dict["Writer parameters"]["EXP_NAME"]
MODEL_VERSION = config_dict["Writer parameters"]["MODEL_VERSION"]
MODEL_NAME = f"efficientnet_{MODEL_VERSION.lower()}"
EXTRA = config_dict["Writer parameters"]["EXTRA"]
MODEL_SAVE_DIR = config_dict["Saving parameters"]["MODEL_SAVE_DIR"]


def main():
    # Set pytorch device. set_device will set "cuda" by default
    device = utils.set_device()

    # Create model object and preset weights and transforms for transfer learning
    weights, model, preprocess = create_effnet.create_efficientnet_model(model_version=MODEL_VERSION, device=device)

    data_path = Path("data/")
    image_path = data_path / "TES"
    # train_dir = image_path / "train"
    # test_dir = image_path / "test"

    # Recreate preset normalize and transforms to control resize and crop values.
    my_normalize = transforms.Normalize(
        mean=preprocess.mean,
        std=preprocess.std,
    )
    my_transform = transforms.Compose([
        transforms.Resize(size=preprocess.resize_size[0], interpolation=preprocess.interpolation, antialias=True),
        transforms.ToTensor(),
        my_normalize
    ])

    # Get dataloaders and the list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_datasets_from_classes_in_folders(
        data_dir=str(image_path),
        transform=my_transform,
        batch_size=BATCH_SIZE)

    # Freeze model layers
    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds()

    num_classes = len(class_names)
    # Change classifier to have number of output layers equal to number of classes
    model.classifier = create_effnet.change_classifier(in_features=1280, out_features=num_classes, device=device)

    # Set up loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    utils.set_seeds()  # Set random seed to fixed value for reproducibility
    start_time = timer()  # Measure time of training
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        writer=utils.create_writer(experiment_name=EXP_NAME, model_name=MODEL_NAME),
    )
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Save trained model
    save_filename = f"{MODEL_NAME}_{EXP_NAME}-data_{BATCH_SIZE}-batchsize_{NUM_EPOCHS}-epochs.pth"
    utils.save_model(model=model,
                     target_dir=MODEL_SAVE_DIR,
                     model_name=save_filename)


if __name__ == "__main__":
    if data_setup.NUM_WORKERS != 0:
        multiprocessing.freeze_support()  # Required for multiprocessing to work
    main()
