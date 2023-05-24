import yaml
import torch
import torchvision
from pathlib import Path

import object_creator
import utils
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = utils.set_device()

# Open config files and set variables
CONFIG_FILENAME = "config.yaml"
SETTINGS_FILENAME = "settings.yaml"
with open(SETTINGS_FILENAME, "r") as stream:
    settings_dict = yaml.safe_load(stream)
with open(CONFIG_FILENAME, "r") as stream:
    config_dict = yaml.safe_load(stream)
MODEL_VERSION = config_dict["MODEL_VERSION"]
DATA_DIR = config_dict["DATA_DIR"]
MODEL_PATH = settings_dict["model_path"]
image_path = settings_dict["img_path"]
img_path = Path(DATA_DIR)
img_path = img_path / "TES"


def main():
    # Create a model in which state dict will be loaded
    weights, model, preprocess = data_setup.create_efficientnet_model(MODEL_VERSION, device)

    # Recreate preset transforms manually to control resize and crop sizes
    my_normalize = transforms.Normalize(
        mean=preprocess.mean,
        std=preprocess.std,
    )
    my_transform = transforms.Compose([
        transforms.Resize(size=preprocess.resize_size[0], interpolation=preprocess.interpolation, antialias=True),
        transforms.ToTensor(),
        my_normalize
    ])

    # Get a list of class names from names of image folders
    class_names = torchvision.datasets.ImageFolder(str(img_path)).classes

    # Change classifier to have number of output layers equal to number of classes, then load state dict of saved model
    model.classifier = data_setup.change_classifier(in_features=1280, out_features=len(class_names), device=device)
    model.load_state_dict(torch.load(MODEL_PATH))

    img = Image.open(image_path)  # Create PIL object of the image

    model.eval()  # Turn on model evaluation mode and inference mode
    with torch.inference_mode():
        transformed_image = my_transform(img).unsqueeze(dim=0)  # Apply transforms
        target_image_pred = model(transformed_image.to(device))  # Get a tensor of logits (non-normalized predictions)

    # Convert logits to prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    probs_list = target_image_pred_probs.tolist()[0]
    output_string = ""
    for i in range(len(class_names)):
        output_string += f"{class_names[i]}: {probs_list[i] * 100:.2f}%    "
    plt.figure()
    plt.imshow(img)
    plt.title(output_string)
    plt.axis(False)
    plt.show()


if __name__ == "__main__":
    main()
