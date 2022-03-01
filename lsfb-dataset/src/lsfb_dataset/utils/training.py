"""
@author ppoitier
@version 1.0
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, Optional
import copy
import time
from tqdm.auto import tqdm
import numpy as np
import logging

from lsfb_dataset.utils.metrics import ClassifierMetrics


def train_rnn_model(
        model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
        data_loaders: Dict[str, DataLoader],
        scheduler=None, num_epochs=5, num_classes=2,
        progress_bar=True
):
    """
    Train a recurrent neural network like an LSTM using the PyTorch library.

    Parameters
    ----------
    model : nn.Module
        The model that is trained.
    criterion : nn.Module
        The loss function used in the training.
    optimizer : torch.optim.Optimizer
        The optimizer for back-propagation.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training the model.
        The {data_loaders} dictionary must have 'train' and 'val' keys !
    scheduler : Optional[Object]
        Scheduler that update the learning rate. It is not necessary.
    num_epochs : int
        The number of epochs before the training completion.
    num_classes : int
        The number of classes of the items.
    progress_bar : bool
        If true, show a progress bar. Otherwise, it does not.

    Returns
    -------
    nn.Module
        The model with the best balanced accuracy.
    ClassifierMetrics
        The metrics about the training phases
    ClassifierMetrics
        The metrics about the validation phases
    """
    since = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    criterion = criterion.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_metrics = ClassifierMetrics(num_classes=num_classes)
    val_metrics = ClassifierMetrics(num_classes=num_classes)

    np.set_printoptions(precision=4)

    for epoch in range(1, num_epochs + 1):
        logging.info(f'{"-"*10} EPOCH {epoch}')
        epoch_time_elapsed = time.time()
        epoch_loss = 0
        epoch_batches = 0

        logging.info('Training model...')
        for features, targets in tqdm(data_loaders['train'], disable=(not progress_bar)):
            features = features.to(device).float()
            targets = targets
            # features : (batch_size, seq_len, features_nb)
            # targets  : (batch_size, seq_len)

            # --- forward
            scores = model(features)
            _, pred = torch.max(scores, 2)
            loss = criterion(scores.view(-1, num_classes), targets.to(device).view(-1))

            train_metrics.add_predictions(targets, pred.cpu())

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        train_metrics.add_loss(epoch_loss/epoch_batches)
        train_metrics.commit()

        logging.info(f'Training: loss = {train_metrics.loss:.4f} ; accuracy = {train_metrics.accuracy:.4f}')
        logging.info(f'Recall: {train_metrics.recall()}')
        logging.info(f'Balanced accuracy: {train_metrics.balanced_accuracy:.4f}')

        epoch_loss = 0
        epoch_batches = 0

        logging.info('Validation...')
        for features, targets in tqdm(data_loaders['val'], disable=(not progress_bar)):
            features = features.to(device).float()
            targets = targets
            # features : (batch_size, seq_len, features_nb)
            # targets  : (batch_size, seq_len)

            # forward
            with torch.no_grad():
                scores = model(features)
                _, pred = torch.max(scores, 2)
                loss = criterion(scores.view(-1, num_classes), targets.to(device).view(-1))

            epoch_loss += loss.item()
            epoch_batches += 1

            pred = pred.cpu()
            val_metrics.add_predictions(targets, pred)
            val_metrics.add_duration_distributions(targets, pred)
            val_metrics.add_transition_distributions(targets, pred)

        val_metrics.add_loss(epoch_loss/epoch_batches)
        val_metrics.commit()

        # noinspection PyUnboundLocalVariable
        val_metrics.add_roc_curve(
            targets.view(-1).numpy(),
            scores.detach().view(-1, num_classes).cpu().numpy()
        )

        epoch_balanced_acc = val_metrics.balanced_accuracy
        epoch_time_elapsed = time.time() - epoch_time_elapsed

        logging.info(f'Epoch complete in {epoch_time_elapsed // 60}min {epoch_time_elapsed % 60:.0f}s')
        logging.info(f'Validation: loss = {val_metrics.loss:.4f} ; accuracy = {val_metrics.accuracy:.4f}')
        logging.info(f'Recall: {val_metrics.recall()}')
        logging.info(f'Balanced accuracy: {epoch_balanced_acc:.4f}')

        if scheduler is not None:
            # noinspection PyUnresolvedReferences
            scheduler.step()

        if epoch_balanced_acc > best_acc:
            best_acc = epoch_balanced_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}s')
    logging.info(f'Best balanced accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    model.eval()
    return model, (train_metrics, val_metrics)
