from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_scores = []
    validation_scored = []
    for i in range(cv):
        validation_idxs = np.arange(X.shape[0])
        validation_idxs = np.where(validation_idxs % cv == i, 1, 0)
        validation_set, validation_labels = X[validation_idxs == 1], y[validation_idxs == 1]
        train_set, train_labels = X[validation_idxs == 0], y[validation_idxs == 0]
        estimator.fit(train_set, train_labels)
        train_pred = estimator.predict(train_set)
        val_pred = estimator.predict(validation_set)
        train_scores.append(scoring(train_labels, train_pred))
        validation_scored.append(scoring(validation_labels, val_pred))
    return np.mean(train_scores), np.mean(validation_scored)


