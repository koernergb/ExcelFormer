import enum
from typing import Any, Optional, Union, cast, Tuple, Dict

import numpy as np
import scipy.special
import sklearn.metrics as skm

from . import util
from .util import TaskType


class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'


def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        if task_type == TaskType.BINCLASS:
            probs = scipy.special.softmax(y_pred, axis=1)  # Convert to probabilities
            labels = probs.argmax(axis=1)  # Take highest probability class
            probs = probs[:, 1]  # Only keep positive class probability for ROC AUC
        else:
            probs = scipy.special.softmax(y_pred, axis=1)
            labels = probs.argmax(axis=1)
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
        labels = probs.argmax(axis=1)
        if task_type == TaskType.BINCLASS:
            probs = probs[:, 1]  # Only keep positive class probability for ROC AUC
    else:
        util.raise_unknown('prediction_type', prediction_type)

    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)


    # Add these debug prints
    # print("y_true shape:", y_true.shape)
    # print("y_true sample:", y_true[:5])
    # print("y_pred shape:", y_pred.shape)
    # print("y_pred sample:", y_pred[:5])
    #print("prediction_type:", prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        result = {'rmse': rmse}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        
        # Add these debug prints too
        # print("labels shape:", labels.shape)
        #print("labels sample:", labels[:5])
        
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result
