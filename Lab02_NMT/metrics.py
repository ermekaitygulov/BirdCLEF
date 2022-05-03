import numpy as np
from sklearn.metrics import (
    f1_score,
    multilabel_confusion_matrix,
)

from utils import add_to_catalog

METRICS_CATALOG = {
    'f1': f1_score
}


@add_to_catalog('comp_metric', METRICS_CATALOG)
def comp_metric(y_true, y_pred, epsilon=1e-9):
    """ Function to calculate competition metric in an sklearn like fashion

    Args:
        y_true{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Ground truth (correct) target values.
        y_pred{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Estimated targets as returned by a classifier.
    Returns:
        The single calculated score representative of this competitions evaluation
    """

    # Get representative confusion matrices for each label
    mlbl_cms = multilabel_confusion_matrix(y_true, y_pred)

    # Get two scores (TP and TN SCORES)
    tp_scores = np.array([
        mlbl_cm[1, 1]/(epsilon+mlbl_cm[:, 1].sum()) \
        for mlbl_cm in mlbl_cms
        ])
    tn_scores = np.array([
        mlbl_cm[0, 0]/(epsilon+mlbl_cm[:, 0].sum()) \
        for mlbl_cm in mlbl_cms
        ])

    # Get average
    tp_mean = tp_scores.mean()
    tn_mean = tn_scores.mean()

    return round((tp_mean+tn_mean)/2, 8)


@add_to_catalog('balanced_accuracy', METRICS_CATALOG)
def balanced_accuracy(pred, target, eps=1e-6):
    tp = (pred * target).sum(axis=-1)
    fn = ((1 - pred) * target).sum(axis=-1)
    fp = (pred * (1 - target)).sum(axis=-1)
    tn = ((1 - pred) * (1 - target)).sum(axis=-1)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    return (0.5 * (tpr + tnr)).mean()