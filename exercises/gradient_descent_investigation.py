import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
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

        values.append(output)
        parameters.append(weight)


    return func, values, parameters



def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    norm_fig1 = go.Figure(go.Scatter(), layout=dict(title="L1 norm value as a function of the GD iteration number"))
    norm_fig2 = go.Figure(go.Scatter(), layout=dict(title="L2 norm value as a function of the GD iteration number"))
    for eta in etas:
        learning_rate = FixedLR(eta)
        l1_module = L1(init.copy())

        l1_func, l1_values, l1_parameters = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate, callback=l1_func)
        gd.fit(l1_module, None, None)
        fig = plot_descent_path(L1,
                                 np.array(l1_parameters),
                                 title=f"Module: L1, eta={eta}")
        fig.write_html('tmp.html', auto_open=True)

        norm_fig1.add_trace(go.Scatter(x=np.arange(1, 1001),
                                 y=np.array(l1_values),
                                 name=f"eta={eta}"))
        # print(f"L1 best loss value with eta={eta} is {np.min(np.array(l1_values)[:, 0])}")

        l2_module = L2(init.copy())
        l2_func, l2_values, l2_parameters = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate, callback=l2_func)
        gd.fit(l2_module, None, None)
        fig1 = plot_descent_path(L2,
                                 np.array(l2_parameters),
                                 title=f"Module: L2, eta={eta}")
        fig1.write_html('tmp.html', auto_open=True)

        norm_fig2.add_trace(go.Scatter(x=np.arange(1, 1001),
                                 y=np.array(l2_values),
                                 name=f"eta={eta}"))
        # print(f"L2 best loss value with eta={eta} is {np.min(np.array(l2_values)[:, 0])}")


    norm_fig1.write_html('tmp.html', auto_open=True)
    norm_fig2.write_html('tmp1.html', auto_open=True)





def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):

    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(go.Scatter(), layout=dict(title="L1 module  convergence rate as function of iteration number"))

    for gamma in gammas:
        exp_learning_rate = ExponentialLR(eta, gamma)
        l1_module = L1(init.copy())
        l1_func, l1_values, l1_parameters = get_gd_state_recorder_callback()
        exp_gd = GradientDescent(exp_learning_rate, callback=l1_func)
        exp_gd.fit(l1_module, None, None)

        # Plot algorithm's convergence for the different values of gamma

        fig.add_trace(go.Scatter(x=np.arange(1, 1001),
                                 y=np.array(l1_values),
                                 name=f"gamma={gamma}"))

        # print(f"L1 best loss value with gamma={gamma} is {np.min(np.array(l1_values))}")

    # Plot descent path for gamma=0.95
        if gamma == 0.95:
            fig2 = plot_descent_path(L1,
                                     np.array(l1_parameters),
                                     title=f"L1, gamma:{gamma}")
            fig2.write_html('tmp.html', auto_open=True)

    fig.write_html('tmp1.html', auto_open=True)



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
    callback, descent_path, values = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback, learning_rate=FixedLR(1e-4), max_iter=20000)
    lr = LogisticRegression(solver=gd)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_train)

    # Plotting convergence rate of logistic regression over SA heart disease data
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    roc = tpr - fpr
    best_roc_idx = np.argmax(roc)
    a_star = thresholds[best_roc_idx]
    # print(f"The optimal alpha value is: {a_star}")

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=f"ROC Curve Of Fitted Model - AUC={auc(fpr, tpr):.6f}",
            xaxis=dict(title="False Positive Rate (FPR)"),
            yaxis=dict(title="True Positive Rate (TPR)}")))\
        .write_html('tmp.html', auto_open=True)

    lr = LogisticRegression(solver=gd, alpha=a_star)
    lr.fit(X_train, y_train)
    # print(f"The test error using the optimal alpha is: {lr.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ["l1", "l2"]:
        train_err, val_err = [], []
        for lam in lambdas:
            model = LogisticRegression(solver=gd, penalty=penalty, lam=lam)
            t_err, v_arr = cross_validate(model, X_train, y_train, misclassification_error)
            train_err.append(t_err)
            val_err.append(v_arr)

        best_lam = lambdas[np.argmin(val_err)]
        # print(f"The best lambda value for {penalty} regularization is: ", best_lam)
        model = LogisticRegression(solver=gd, penalty=penalty, lam=best_lam)
        model.fit(X_train, y_train)
        # print(f"The test error using the optimal lambda with {penalty} regularization is: {model.loss(X_test, y_test)}")







if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
