"""
Contains functionality for creating PyTorch DataLoader instances for image
classification data.
"""
import pathlib

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

NUM_WORKERS = 0
BATCH_SIZE = 32
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def _instantiate_dataloader(
        path: str | pathlib.Path,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
        shuffle: bool
) -> Tuple[DataLoader, None, List[str]]:
    data = ImageFolder(root=path, transform=transform)
    return DataLoader(dataset=data,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=True), None, data.classes

def create_dataloaders(
    train_dir: Optional[str | pathlib.Path] = None,
    test_dir: Optional[str | pathlib.Path] = None,
    train_transform: Optional[transforms.Compose] = STANDARD_TRANSFORM,
    test_transform: Optional[transforms.Compose] = None,
    batch_size: Optional[int] = BATCH_SIZE,
    num_workers: Optional[int] = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders.

    Takes in a training and testing directory and turns them into PyTorch
    Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: Instance of torchvision.transforms used to perform
            transformations on the training data. By default a chain of
            Resize (with size of 244, 244) and ToTensor transformations is
            used. 
        test_transform: Instance of torchvision.transforms used to perform
            transformations on the testing data. If None the train_transform
            will be used instead.
        batch_size: Number of samples per batch in each of the DataLoaders
            instances.
        num_workers: An integer for number of threads that will extract the
            batches from the DataLoaders instances.
            
    Returns:
        A tuple conformed of the training and testing dataloader instances and
        a list of the target classes founded in the given data.
        If both train_dir and test_dir are empty, None will be returned.
        If train_dir is not empty but test_dir is, only the train_dataloader
        will be returned.
        If test_dir is not empty but train_dir is, only the test_dataloader
        will be returned.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir='train_dir',
                test_dir='test_dir',
                train_transform=transforms.Resize(),
                batch_size=32,
                num_workers=4
            )
    """
    if not test_transform:
        test_transform = train_transform

    if not train_dir and not test_dir:
        return None
    elif train_dir and not test_dir:
        return _instantiate_dataloader(train_dir,
                                       train_transform,
                                       batch_size,
                                       num_workers,
                                       True)
    elif not train_dir and test_dir:
        return _instantiate_dataloader(test_dir,
                                       test_transform,
                                       batch_size,
                                       num_workers,
                                       False)

    train_dataloader, _, classes = _instantiate_dataloader(train_dir,
                                                           train_transform,
                                                           batch_size,
                                                           num_workers,
                                                           True)
    test_dataloader, _, _ = _instantiate_dataloader(test_dir,
                                                    test_transform,
                                                    batch_size,
                                                    num_workers,
                                                    False)

    return train_dataloader, test_dataloader, classes
