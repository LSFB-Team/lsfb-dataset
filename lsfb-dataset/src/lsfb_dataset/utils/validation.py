import logging

import pandas as pd
import torch
from tqdm.auto import tqdm

from lsfb_dataset.utils.metrics import ClassifierMetrics, \
    compute_accuracy_from_conf_matrix, compute_balanced_accuracy_from_conf_matrix


def evaluate_rnn_model(model, data_loader, num_classes, progress_bar=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    metrics = ClassifierMetrics(num_classes)

    epoch_batches = 0

    logging.info('Evaluating model...')

    records = []

    with torch.no_grad():
        for index, (features, targets) in enumerate(tqdm(data_loader, disable=(not progress_bar))):
            features = features.to(device)

            batch_size = targets.size(0)

            scores = model(features)
            _, pred = torch.max(scores, 2)

            pred = pred.cpu()
            metrics.add_predictions(targets, pred)
            metrics.add_transitions_matrix(pred)
            metrics.add_duration_distributions(targets, pred)
            metrics.add_transition_distributions(targets, pred)

            records.append((
                compute_accuracy_from_conf_matrix(metrics.current_conf),
                compute_balanced_accuracy_from_conf_matrix(metrics.current_conf),
            ))

            epoch_batches += batch_size

    metrics.commit()

    logging.info(f'Accuracy: {metrics.accuracy()}')
    logging.info(f'Balanced accuracy: {metrics.balanced_accuracy()}')
    logging.info(f'Recall: {metrics.recall()}')

    df_records = pd.DataFrame.from_records(data=records, columns=['accuracy', 'balanced_accuracy'])

    return metrics, df_records


def make_test_prediction(model, features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    features = features.to(device)

    with torch.no_grad():
        scores = model(features.unsqueeze(1))
        _, pred = torch.max(scores, 2)

    return scores.squeeze().cpu(), pred.squeeze().cpu()
