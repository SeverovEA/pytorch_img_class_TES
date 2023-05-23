import torchvision.models
import torch
from typing import Tuple


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


def change_classifier(in_features: int,
                      out_features: int,
                      device: str,
                      dropout_rate: float = 0.2
                      ):
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_rate, inplace=True),
        torch.nn.Linear(in_features=in_features,
                        out_features=out_features,
                        bias=True)).to(device)
    return classifier
