import multiprocessing
from pathlib import Path
from timeit import default_timer as timer
import torch
from torch import nn
from torchvision import transforms
import object_creator
import engine
import utils

# Open config files and set variables
config_dict, settings_dict = utils.yamls_to_dict(("config.yaml", "settings.yaml"))
NUM_EPOCHS = int(config_dict["NUM_EPOCHS"])
LEARNING_RATE = int(config_dict["LEARNING_RATE"])
BATCH_SIZE = int(config_dict["BATCH_SIZE"])
EXP_NAME = config_dict["EXP_NAME"]
MODEL_VERSION = config_dict["MODEL_VERSION"]
MODEL_NAME = f"efficientnet_{MODEL_VERSION.lower()}"
EXTRA = config_dict["EXTRA"]
MODEL_SAVE_DIR = config_dict["MODEL_SAVE_DIR"]


def main():
    # Set pytorch device. set_device will set "cuda" by default
    device = utils.set_device()

    # Create model object and preset weights and transforms for transfer learning
    weights, model, preprocess = object_creator.create_efficientnet_model(model_version=MODEL_VERSION, device=device)

    data_path = Path("data/")
    image_path = data_path / "TES"

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
    train_dataloader, test_dataloader, class_names = object_creator.create_datasets_from_classes_in_folders(
        data_dir=str(image_path),
        transform=my_transform,
        batch_size=BATCH_SIZE)

    # Freeze model layers
    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds()

    num_classes = len(class_names)
    # Change classifier to have number of output layers equal to number of classes
    model.classifier = utils.change_classifier(in_features=1280, out_features=num_classes, device=device)

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
        writer=object_creator.create_writer(experiment_name=EXP_NAME, model_name=MODEL_NAME),
    )
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Save trained model
    save_filename = f"{MODEL_NAME}_{EXP_NAME}-data_{BATCH_SIZE}-batchsize_{NUM_EPOCHS}-epochs.pth"
    utils.save_model(model=model,
                     target_dir=MODEL_SAVE_DIR,
                     model_name=save_filename)


if __name__ == "__main__":
    if object_creator.NUM_WORKERS != 0:
        multiprocessing.freeze_support()  # Required for multiprocessing to work
    main()
