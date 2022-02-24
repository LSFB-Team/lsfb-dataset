import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from ..utils.annotations import get_annotations_durations, create_coerc_vec


def compute_accuracy(conf):
    return np.trace(conf) / conf.sum()


def plot_distributions(dist_true, dist_pred, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.hist(dist_true, bins=100, range=(0, 5000), alpha=0.4, density=True, label='Target')
    plt.hist(dist_pred, bins=100, range=(0, 5000), alpha=0.4, density=True, label='Prediction')
    plt.legend()
    plt.show()


class ClassifierMetrics:
    def __init__(self, num_classes=2, labels=None):
        self.num_classes = num_classes

        if labels is None:
            labels = range(num_classes)
        else:
            assert len(labels) == num_classes,\
                'The number of labels must be the same as the number of classes'

        self.labels = labels

        self.loss_evolution = []
        self.accuracy_evolution = []
        self.true_duration = []
        self.pred_duration = []
        self.true_transitions = []
        self.pred_transitions = []
        self.roc_curves = []

        self.confs = []

        self.current_conf = None
        self.current_true_durations = []
        self.current_pred_durations = []
        self.current_true_transitions = []
        self.current_pred_transitions = []

    @property
    def conf(self):
        return self.confs[-1]

    @property
    def loss(self):
        return self.loss_evolution[-1]

    @property
    def accuracy(self):
        return self.accuracy_evolution[-1]

    @property
    def recall(self):
        conf = self.conf
        recall = []
        for c in range(self.num_classes):
            recall.append(conf[c, c] / conf[c, :].sum())
        return np.array(recall)

    @property
    def balanced_accuracy(self):
        return self.recall.sum() / self.num_classes

    @property
    def roc_curve(self):
        return self.roc_curves[-1]

    @property
    def roc_auc(self):
        curves = self.roc_curve

        auc_scores = []
        for c in range(len(curves)):
            fpr, tpr, _ = curves[c]
            auc_scores.append(auc(fpr, tpr))

        return auc_scores

    def add_predictions(self, y_true, y_pred):
        if self.current_conf is None:
            self.current_conf = np.zeros((self.num_classes, self.num_classes))
        assert y_true.shape == y_pred.shape, 'Targets and predictions have different shapes.'

        for idx in range(y_true.shape[0]):
            self.current_conf += confusion_matrix(y_true[idx], y_pred[idx], labels=self.labels)

    def add_duration_distributions(self, y_true, y_pred):
        for idx in range(y_true.shape[0]):
            self.current_true_durations += get_annotations_durations(y_true[idx])
            self.current_pred_durations += get_annotations_durations(y_pred[idx] > 0)

    def add_transition_distributions(self, y_true, y_pred):
        assert self.num_classes == 2 or self.num_classes == 3,\
            'Wrong classes number for transitions.'

        for idx in range(y_true.shape[0]):
            annot_true = y_true[idx]
            annot_pred = y_pred[idx]

            if self.num_classes == 2:
                annot_true = create_coerc_vec(annot_true)
                annot_pred = create_coerc_vec(annot_pred > 0)

            self.current_true_transitions += get_annotations_durations(annot_true, value=2)
            self.current_pred_transitions += get_annotations_durations(annot_pred, value=2)

    def add_roc_curve(self, y_true, y_score):
        curves = []
        for c in range(self.num_classes):
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, c], pos_label=c)
            curves.append((fpr, tpr, thresholds))
        self.roc_curves.append(curves)

    def add_loss(self, loss):
        self.loss_evolution.append(loss)

    def commit(self):
        self.accuracy_evolution.append(compute_accuracy(self.current_conf))
        self.confs.append(self.current_conf)
        self.current_conf = None

        if len(self.current_true_durations) > 0:
            self.true_duration.append(self.current_true_durations)
            self.pred_duration.append(self.current_pred_durations)
            self.true_transitions.append(self.current_true_transitions)
            self.pred_transitions.append(self.current_pred_transitions)
            self.current_true_durations = []
            self.current_pred_durations = []
            self.current_true_transitions = []
            self.current_pred_transitions = []

    def plot_conf(self):
        plt.figure()
        plt.title('Confusion matrix')
        sn.heatmap(self.confs[-1], annot=True, fmt='.0f', cmap='flare')
        plt.show()

    def plot_duration_distributions(self):
        plot_distributions(self.true_duration[-1], self.pred_duration[-1], 'Annotation duration distributions')

    def plot_transition_distributions(self):
        plot_distributions(self.true_transitions[-1], self.pred_transitions[-1],
                           'Annotation transition duration distributions')

    def plot_roc_curve(self):
        curves = self.roc_curve
        roc_auc = self.roc_auc

        fig, ax1 = plt.subplots()
        ax1.set_title('Roc curve')
        for c in range(len(curves)):
            fpr, tpr, _ = curves[c]
            score = roc_auc[c]
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=score)
            display.plot(ax=ax1, label=f'class {c}')
        plt.show()

