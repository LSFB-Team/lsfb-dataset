import logging

import torch
from tqdm.auto import tqdm

from lsfb_dataset.utils.metrics import ClassifierMetrics


def evaluate_rnn_model(model, data_loader, num_classes, progress_bar=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    metrics = ClassifierMetrics(num_classes)

    epoch_batches = 0

    logging.info('Evaluating model...')

    with torch.no_grad():
        for features, targets in tqdm(data_loader, disable=(not progress_bar)):
            features = features.to(device)

            batch_size = targets.size(0)

            scores = model(features)
            _, pred = torch.max(scores, 2)

            pred = pred.cpu()
            metrics.add_predictions(targets, pred)
            metrics.add_duration_distributions(targets, pred)
            metrics.add_transition_distributions(targets, pred)

            epoch_batches += batch_size

    metrics.commit()

    logging.info(f'Accuracy: {metrics.accuracy}')
    logging.info(f'Balanced accuracy: {metrics.balanced_accuracy}')
    logging.info(f'Recall: {metrics.recall()}')

    return metrics


def make_test_prediction(model, features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    features = features.to(device)

    with torch.no_grad():
        scores = model(features.unsqueeze(1))
        _, pred = torch.max(scores, 2)

    return scores.squeeze().cpu(), pred.squeeze().cpu()
