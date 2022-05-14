import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_err = []
    test_err = []
    for t in range(1, n_learners+1):
        train_err.append(adaboost.partial_loss(train_X, train_y, t))
        test_err.append(adaboost.partial_loss(test_X, test_y, t))

    fig = go.Figure(go.Scatter(x=np.arange(1, n_learners+1), y=train_err, name="Train Error"))
    fig.add_trace(go.Scatter(y=test_err, name="Test Error"))
    fig.update_layout(
        title=f"the training- and test errors as a function of the number of fitted learners, noise level = {noise}",
        margin=dict(t=100))
    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    model_names = ["5 Iterations", "50 Iterations",
                   "100 Iterations", "250 Iterations"]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=0.1, vertical_spacing=.1)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X,t), lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title="Decision Surface using different number of stump ensemble",
        margin=dict(t=100))

    fig.show()


    # Question 3: Decision surface of best performing ensemble
    errors = []
    for t in range(n_learners+1):
        errors.append(adaboost.partial_loss(test_X, test_y, t))

    best_ens = np.argmin(errors)
    fig1 = go.Figure(data=[decision_surface(lambda X: adaboost.partial_predict(X,best_ens), lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    fig1.update_layout(
        title=f"Decision Surface of the best ensemble - {best_ens}, "
              f"Acc={accuracy(adaboost.partial_predict(test_X, best_ens), test_y)}",
        margin=dict(t=100))
    fig1.show()


    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig2 = go.Figure(data=[decision_surface(adaboost.predict, lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=train_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1),
                                               size=D))])
    fig2.update_layout(
        title=f"Decision Surface according to 250 stumps ensemble, "
              f"size of markers according to sample weight. "
              f"Noise Level = {noise}",
        margin=dict(t=100))
    fig2.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # fit_and_evaluate_adaboost(0.4)
