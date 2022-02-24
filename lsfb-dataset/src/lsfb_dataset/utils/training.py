import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from typing import Dict
import copy
import time
from tqdm.auto import tqdm
import numpy as np

from lsfb_dataset.utils.metrics import ClassifierMetrics


def train_rnn_model(
        model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
        data_loaders: Dict[str, DataLoader],
        scheduler=None, num_epochs=5, num_classes=2,
        progress_bar=True
):
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
        print('-' * 10, 'EPOCH', epoch)

        print('Training model...')
        for features, targets in tqdm(data_loaders['train'], disable=(not progress_bar)):
            features = features.to(device).squeeze(1).float()
            encoded_targets = one_hot(targets.to(device).squeeze().long(), num_classes=num_classes).float()

            # forward
            scores = model(features)
            _, pred = torch.max(scores, 2)
            loss = criterion(scores.squeeze(), encoded_targets)

            train_metrics.add_predictions(targets, pred.cpu())
            train_metrics.add_loss(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

        train_metrics.commit()
        print(f'Training: loss = {train_metrics.loss:.4f} ; accuracy = {train_metrics.accuracy:.4f}')
        print(f'Recall: {train_metrics.recall}')
        print(f'Balanced accuracy: {train_metrics.balanced_accuracy:.4f}')

        print('Validation...')
        for features, targets in tqdm(data_loaders['val'], disable=(not progress_bar)):
            features = features.to(device).squeeze(1).float()
            encoded_targets = one_hot(targets.to(device).squeeze().long(), num_classes=num_classes).float()

            # forward
            scores = model(features)
            _, pred = torch.max(scores, 2)
            loss = criterion(scores.squeeze(), encoded_targets)

            pred = pred.cpu()

            val_metrics.add_predictions(targets, pred)
            val_metrics.add_duration_distributions(targets, pred)
            val_metrics.add_transition_distributions(targets, pred)
            val_metrics.add_loss(loss.item())

        val_metrics.commit()
        val_metrics.add_roc_curve(
            targets.squeeze().numpy(),
            scores.squeeze().detach().cpu().numpy()
        )

        epoch_balanced_acc = val_metrics.balanced_accuracy

        print(f'Validation: loss = {val_metrics.loss:.4f} ; accuracy = {val_metrics.accuracy:.4f}')
        print(f'Recall: {val_metrics.recall}')
        print(f'Balanced accuracy: {epoch_balanced_acc:.4f}')
        val_metrics.plot_conf()
        val_metrics.plot_duration_distributions()
        val_metrics.plot_transition_distributions()
        val_metrics.plot_roc_curve()

        if scheduler is not None:
            scheduler.step()

        if epoch_balanced_acc > best_acc:
            best_acc = epoch_balanced_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print()
    print(f'Training complete in {time_elapsed // 60}min {time_elapsed % 60:.0f}s')
    print(f'Best balanced accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    model.eval()
    return model, (train_metrics, val_metrics)