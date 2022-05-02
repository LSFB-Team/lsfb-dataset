import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import path
import os
import matplotlib.pyplot as plt
import pickle

from lsfb_dataset.utils.metrics import ClassifierMetrics, VideoSegmentationRecords
from lsfb_dataset.datasets.lsfb_cont.skeleton_landmarks import SkeletonLandmarksDataset


def evaluate_rnn_model(
        model,
        data_loader,
        num_classes: int,
        model_name: str,
        dest_dir=None,
        progress_bar=True,
        plots_dir=None,
        predictions_dir=None,
        video_names=None,
):
    assert 0 < num_classes <= 3, 'Unsupported number of classes.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    plots_dir = path.join(plots_dir, model_name)
    os.makedirs(plots_dir, exist_ok=True)

    metrics = ClassifierMetrics(num_classes)
    recorder = VideoSegmentationRecords(
        model_name,
        isolate_transitions=(num_classes == 3),
        video_names=video_names,
        plots_dir=plots_dir,
    )

    epoch_batches = 0

    logging.info('Evaluating model...')

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

            targets = targets[0]
            pred = pred[0]

            if num_classes == 2:
                likelihood = torch.sigmoid(scores[0]).squeeze().cpu()
            else:
                likelihood = torch.softmax(scores[0], dim=1).squeeze().cpu()

            recorder.add_record(targets, pred, likelihood[:, 1])

            if predictions_dir is not None:
                recorder.add_prediction(targets, likelihood)

            epoch_batches += batch_size

    metrics.commit()
    records = recorder.get_records()

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        with open(os.path.join(predictions_dir, f'{model_name}.preds'), 'wb') as file:
            pickle.dump(recorder.get_likelihoods(), file)

    logging.info(f'Accuracy: {metrics.accuracy()}')
    logging.info(f'Balanced accuracy: {metrics.balanced_accuracy()}')
    logging.info(f'Recall: {metrics.recall()}')

    if dest_dir is not None:
        torch.save(metrics.state_dict(), path.join(dest_dir, f'{model_name}.metrics'))
        records.to_csv(path.join(dest_dir, f'records_{model_name}.csv'), index=False)

    return metrics, records


def make_test_prediction(model, root, df_video, class_nb=2):
    dataset = SkeletonLandmarksDataset(root, df_video, isolate_transitions=(class_nb == 3))
    dataloader = DataLoader(dataset, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)

    features, targets = next(iter(dataloader))
    features = features.to(device)

    with torch.no_grad():
        scores = model(features)
        _, pred = torch.max(scores, 2)

    targets = targets[0].squeeze().numpy()
    probas = torch.sigmoid(scores.squeeze()).cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, (ax0, ax1) = plt.subplots(2, figsize=(40, 8))
    fig.patch.set_facecolor('#cccaca')
    fig.suptitle('Video 0', fontsize=16)

    seq_len = len(targets)
    target_filter = targets == 1
    pred_filter = pred == 1
    acc = np.sum(target_filter == pred_filter) / seq_len
    print(acc)

    talking = np.where(target_filter)[0]
    pred_talking = np.where(pred_filter)[0]

    ax0.set_title('Ground truth')
    ax0.set_ylim(0, 1)
    ax0.set_xlim(0, seq_len)
    ax0.vlines(talking, ymin=0.0, ymax=1.0, label='Annotations')
    ax0.margins(x=0, y=0)
    ax0.get_yaxis().set_visible(False)
    ax0.legend()

    ax1.set_title('Result')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, seq_len)
    ax1.vlines(pred_talking, ymin=0.0, ymax=1.0, color='#ffccab', label='Prediction')
    ax1.plot(probas[:, 1], label='Likelihood')
    ax1.margins(x=0, y=0)
    ax1.legend()

    plt.show()

    return probas, pred
