"""
Contains utility functions used across all the model's life cycle.
"""
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision

from datetime import date
from pathlib import Path
from PIL import Image
from torch.utils import tensorboard
from typing import Optional, Tuple


def estimate_experiment_time(start: float, end: float) -> float:
    """Estimates the difference between start and end time on an experiment.

    Returns:
        A float number representing how many seconds lasted the experiment.
    """
    return end - start


def save_model(model: torch.nn.Module,
               target_dir: Path | str,
               model_name: Path | str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either
            ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                   target_dir="models",
                   model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    model_name = Path(model_name)
    cond = model_name.suffix in (".pth", ".pt")
    if not cond:
        model_name = model_name.with_suffix(".pth")
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def plot_loss_curves(results: pd.DataFrame,
                     model_name: str,
                     save_path: Optional[str | Path] = Path('.'),
                     show: Optional[bool] = True,
                     save_results: Optional[bool] = False):
    """Plots training curves of a results DataFrame.

    Args:
        results: Pandas DataFrame containing the experiments results.
        model_name: Name of the model of which its curves will be plotted.
        save_path: Optional value that will point to the directory in which
            the figure will be saved.
        show: Optional boolean determining whether show the figure or not.
            Useful when saving multiple plots in a loop.
        save_results: Optional boolean determining whether the passed results
            dataframe will be saved in the same location as the save_path
            argument or not.
    """

    # Get the loss values of the results dictionary (training and test)
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the metric values of the results dictionary (training and test)
    train_metric = results['train_metric']
    test_metric = results['test_metric']

    # Figure out how many epochs there were
    epochs = results.query(f"model_name == '{model_name}'").shape[0]
    if epochs == 0:
        raise IndexError("There are no values in the past dataframe with a "
                         f"model named like '{model_name}'.")
    title = f"{model_name} Loss and Metric Curves"
    epochs = range(1, epochs + 1)

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 100)
    plt.legend()

    # Plot metric
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric, label='train_metric')
    plt.plot(epochs, test_metric, label='test_metric')
    plt.title('metric')
    plt.xlabel('Epochs')
    plt.ylim(0, 100)
    plt.legend()

    plt.suptitle(title)

    if save_path:
        save_path = Path(save_path).joinpath(title).with_suffix('.png')
        plt.savefig(save_path)

    if show:
        plt.show()

    if save_results:
        save_path = save_path.parent / f'{title}.csv'
        results.to_csv(save_path)


def create_writer(
        experiment_name: str,
        model_name: str,
        extra: Optional[str] = '',
        log_dir: Optional[Path] = Path('.')
) -> tensorboard.writer.SummaryWriter:
    """Creates a SummaryWriter instance that will track the ran experiment.

    SummaryWriter belongs to the torch.utils.tensorboard module. This
    instance will track the experiment that an specific model will run through.

    Args:
        experiment_name: String that represents the name of the experiment.
        model_name: String that represents the name of the model that will
            be trained.
        extra: String that represents miscellaneous information regarding
            the experiment.
        log_dir: Path that points to where the logs directory will be found.
            By default the current directory will be used.

    Returns:
        A SummaryWriter instance ready to track the experiment findings.
    """
    timestamp = date.today()
    log_dir /= f'runs/{timestamp}/{experiment_name}/{model_name}'
    if extra:
        log_dir /= extra

    return tensorboard.SummaryWriter(log_dir=log_dir)


def transform_image(image_path: Path | str,
                    transform: torchvision.transforms.transforms.Compose,
                    device: torch.device) -> Tuple[Image.Image, torch.Tensor]:
    """Transforms an image found in the given path with the passed transform.

    Returns:
        A tuple containing the opened image and its transformation."""
    with Image.open(Path(image_path)) as image:
        return (
            image,
            transform(image)
            .unsqueeze(dim=0)
            .to(device)
        )
