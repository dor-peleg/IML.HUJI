import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    parameters = []

    def func(output, weight, learning_rate):

        values.append([output, learning_rate])
        parameters.append(weight)


    return func, values, parameters



def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        learning_rate = FixedLR(eta)
        l1_module = L1(init.copy())
        l2_module = L2(init.copy())
        l1_func, l1_values, l1_parameters = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate, callback=l1_func)
        gd.fit(l1_module, None, None)
        fig = plot_descent_path(L1,
                                 np.array(l1_parameters),
                                 title=f"L1, eta:{eta}")
        fig.write_html('tmp.html', auto_open=True)

        l2_func, l2_values, l2_parameters = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate, callback=l2_func)
        gd.fit(l2_module, None, None)
        fig1 = plot_descent_path(L2,
                                 np.array(l2_parameters),
                                 title=f"L2, eta:{eta}")
        fig1.write_html('tmp.html', auto_open=True)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(go.Scatter(), layout=dict(title="Decay rates"))

    for gamma in gammas:
        exp_learning_rate = ExponentialLR(eta, gamma)
        l1_module = L1(init.copy())
        l1_func, l1_values, l1_parameters = get_gd_state_recorder_callback()
        exp_gd = GradientDescent(exp_learning_rate, callback=l1_func)
        exp_gd.fit(l1_module, None, None)

        # Plot algorithm's convergence for the different values of gamma

        fig.add_trace(go.Scatter(x=np.arange(1, 1001),
                                 y=np.array(l1_values)[:, 1],
                                 name=f"gamma={gamma}"))


    # Plot descent path for gamma=0.95
    if gamma == 0.95:
        fig2 = plot_descent_path(L1,
                                 np.array(l1_parameters),
                                 title=f"L1 descent path, gamma:{gamma}")
        fig2.write_html('tmp.html', auto_open=True)

    fig.write_html('tmp.html', auto_open=True)



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    X_train : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    y_train : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    X_test : D
    ataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    y_test : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy().reshape(X_train.shape)
    y_train = y_train.to_numpy().reshape(y_train.shape)
    X_test = X_test.to_numpy().reshape(X_test.shape)
    y_test = y_test.to_numpy().reshape(y_test.shape)
    w = np.ones(X_train.shape[1]) / X_train.shape[1]
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))


    # Plotting convergence rate of logistic regression over SA heart disease data
    # raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
