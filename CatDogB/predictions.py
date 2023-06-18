import matplotlib.pyplot as plt
import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
from utils import transform_image


def predict_plot_image(image_path: Path | str,
                       transform:transforms.transforms.Compose,
                       model: torch.nn.Module,
                       class_names: List[str]) -> Tuple[str, List[float]]:
    """Uses given model to predict and plot the image found in image_path."""
    dev = next(model.parameters()).device
    # If value is str.
    image_path = Path(image_path)
    img, img_t = transform_image(image_path, transform, dev)
    
    model.eval()
    with torch.inference_mode():
        preds = model(img_t).softmax(dim=1)

    pred_argmax = preds.argmax(dim=1)
    max_prob = preds[0][pred_argmax].item()
    pred = class_names[pred_argmax]
    plt.figure(figsize=(12, 6))
    plt.title(f"Predicted Label: {pred.title()} | Pred Prob: {max_prob:.3f}")
    plt.imshow(img)
    plt.axis(False)
    plt.show()
        
    return (pred, preds.squeeze().tolist())
