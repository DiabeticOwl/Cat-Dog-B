"""
Contains functionality for training and evaluating a deep learning model.
"""
import numpy as np
import pandas as pd
import torch
import torchmetrics

from timeit import default_timer as timer
from torch.utils import data, tensorboard
from tqdm.auto import tqdm
from typing import Tuple, Optional
from utils import estimate_experiment_time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training_loop(
    model: torch.nn.Module,
    data_loader: data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric_fn: torchmetrics.Metric,
    device: Optional[torch.device] = DEVICE
) -> Tuple[float, float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        data_loader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        metric_fn: A Metric instance representing the metric function
            that will be used when training to calculate how good
            the model's prediction are.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
        A tuple of training loss, training accuracy metrics and
        an estimate of how long it took to train.
        In the form (train_loss, train_accuracy, estimated_time).
        For example:

        (0.1112, 0.8743, 10.256)
    """
    
    model = model.to(device)
    metric_fn = metric_fn.to(device)
    train_loss, train_metric = 0, 0

    exp_start = timer()
    # Looping through the training batch data.
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        
        model.train()
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # argmax because y_pred are logits.
        train_metric += metric_fn(y_pred.argmax(dim=1), y) * 100
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
    # We already have the sum of train loss per batch.
    # This value will correspond to the average train
    # loss per batch per epoch.
    train_loss /= len(data_loader)
    train_metric /= len(data_loader) 
    
    exp_end = timer()
    return (train_loss.item(),
            train_metric.item(),
            estimate_experiment_time(exp_start, exp_end))


def testing_loop(
    model: torch.nn.Module,
    data_loader: data.DataLoader,
    loss_fn: torch.nn.Module,
    metric_fn: torchmetrics.Metric,
    device: Optional[torch.device] = DEVICE
) -> Tuple[float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        data_loader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        metric_fn: A Metric instance representing the metric function
            that will be used when training to calculate how good
            the model's prediction are.       
        device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
        A tuple of testing loss, testing accuracy metrics and
        an estimate of how long it took to train.
        In the form (test_loss, test_accuracy, estimated_time).
        For example:

        (0.0223, 0.8985, 5.312)
    """
    
    model = model.to(device)
    metric_fn = metric_fn.to(device)
    test_loss, test_metric = 0, 0
    model.eval()
    
    with torch.inference_mode():
        exp_start = timer()
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            test_y_pred = model(X)
        
            test_loss += loss_fn(test_y_pred, y)
            test_metric += metric_fn(test_y_pred.argmax(dim=1), y) * 100
            
        # test loss avg per batch per epoch.
        test_loss /= len(data_loader)
        test_metric /= len(data_loader)
    
    exp_end = timer()
    return (test_loss.item(),
            test_metric.item(),
            estimate_experiment_time(exp_start, exp_end))


def train_model(
    model: torch.nn.Module,
    train_data: data.DataLoader,
    test_data: data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric_fn: torchmetrics.metric.Metric | torch.nn.Module,
    epochs: Optional[int] = 5,
    device: Optional[torch.device] = DEVICE,
    random_seed: Optional[int] = None,
    model_name: Optional[str] = None,
    verbose: Optional[int] = 0,
    writer: Optional[tensorboard.writer.SummaryWriter] = None
) -> pd.DataFrame:
    """Trains the given model and returns its experiment results.

    A training section will be instructed upon the given model using both the
    training and testing data passed. The process will be repeated by epochs
    times, enhancing its ability to learn the patterns that the data has and
    storing the expected evaluation metric results. These experiment results can
    also be tracked using the writer optional parameter.

    Args:
        model: An instance of torch.nn.Module to be subject of learning.
        train_data: The PyTorch DataLoader that will provide the training
            batches of data.
        test_data: The PyTorch DataLoader that will provide the testing batches
            of data.
        loss_fn: An instance of torch.nn.Module that will act as the criterion
            that measures the error the model has in each training section.
        optimizer: An instance of torch.optim.Optimizer that will implement an
            optimization algorithm, updating the model internal parameters.
        metric_fn: A function that will estimate how well the model predictions
            are with respect of the actual values.
        epochs: Number of epochs in which the model will be trained and
            evaluated. By default the model will pass through 5 epochs.
        device: An instance of torch.device that determines the device in which
            the model data and metric algorithms are stored. All computation
            will be executed there.
        random_seed: Represents the random seed that will be used.
        model_name: Represents the name of the model. By default the class name
            will be used.
        verbose: If 0, will print only the first and last epoch information.
            If greater than 0, the value set will be used to determine how much
            epochs will be skipped. Example: With verbose = 5 this function will
            print out each 5 epochs.
        writer: Instance of SummaryWriter used for tracking the experiment.

    Returns:
        A pandas DataFrame containing all the model's experiments results.
    """
    if random_seed:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    
    exps_res = pd.DataFrame([])
    res_dict = {
        "train_loss": [],
        "train_metric": [],
        "train_epoch_runtime": [],
        "test_metric": [],
        "test_loss": [],
        "test_epoch_runtime": []
    }
    
    for epoch in tqdm(range(epochs)):
        (train_loss, train_metric,
         train_epoch_runtime) = training_loop(model,
                                              train_data,
                                              loss_fn,
                                              optimizer,
                                              metric_fn,
                                              device)

        (test_loss, test_metric,
         test_epoch_runtime) = testing_loop(model,
                                            test_data,
                                            loss_fn,
                                            metric_fn,
                                            device)
        
        res_dict['train_loss'].append(train_loss)
        res_dict['train_metric'].append(train_metric)
        res_dict['train_epoch_runtime'].append(train_epoch_runtime)
        res_dict['test_loss'].append(test_loss)
        res_dict['test_metric'].append(test_metric)
        res_dict['test_epoch_runtime'].append(test_epoch_runtime)
       
        if writer:
            writer.add_scalars(main_tag='loss',
                               tag_scalar_dict={'train_loss': train_loss,
                                                'test_loss': test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag='metric',
                               tag_scalar_dict={'train_metric': train_metric,
                                                'test_metric': test_metric},
                               global_step=epoch)
            writer.close()

        if (epoch == 0 or epoch + 1 == epochs or 
            (verbose > 0 and epoch % verbose == 0)):
            print(f"Epoch: {epoch + 1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_metric: {train_metric:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_metric: {test_metric:.4f}")

    exps_res = pd.DataFrame.from_dict(res_dict).reset_index(drop=True)
    if not model_name:
        exps_res['model_name'] = model.__class__.__name__
    else:
        exps_res['model_name'] = model_name
        
    col_order = [
        'model_name',
        'train_loss',
        'train_metric',
        'train_epoch_runtime',
        'test_loss',
        'test_metric',
        'test_epoch_runtime'
    ]
    return exps_res[col_order]
