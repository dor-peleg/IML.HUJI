from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    func = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    samples = np.linspace(-1.2, 2, n_samples)
    labels = func(samples)
    noise = np.random.normal(0, noise, labels.shape)
    noisy_labels = labels + noise
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(samples),
                                                        pd.Series(
                                                            noisy_labels),
                                                        2 / 3)
    train_X = train_X.to_numpy().reshape(train_X.shape[0])
    train_y = train_y.to_numpy().reshape(train_y.shape[0])
    test_X = test_X.to_numpy().reshape(test_X.shape[0])
    test_y = test_y.to_numpy().reshape(test_y.shape[0])

    fig = go.Figure(go.Scatter(x=samples,
                               y=labels,
                               mode="markers",
                               name="True Function"),
                    layout=dict(title="True and Noised Dataset"))
    fig.add_trace(go.Scatter(x=test_X,
                             y=test_y,
                             mode="markers",
                             name="Noisy Test Set"))
    fig.add_trace(go.Scatter(x=train_X,
                             y=train_y,
                             mode="markers",
                             name="Noisy Train Set"))

    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = []
    val_err = []
    sum = []
    for k in range(11):
        model = PolynomialFitting(k)
        avg_train_err, avg_vaL_err = cross_validate(model, train_X, train_y,
                                                    mean_square_error, 5)
        train_err.append(avg_train_err)
        val_err.append(avg_vaL_err)
        sum.append(avg_vaL_err + avg_train_err)

    fig = go.Figure(go.Scatter(x=np.arange(11),
                               y=train_err,
                               name="train error"),
                    layout=dict(title="Train and Validation error as "
                                      "function of polynomial degree"))
    fig.add_trace(go.Scatter(x=np.arange(11),
                             y=val_err,
                             name="validation error"
                             ))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(val_err)
    model = PolynomialFitting(k_star)
    model.fit(train_X, train_y)
    test_pred = model.predict(test_X)
    test_err = mean_square_error(test_pred, test_y)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lams = np.linspace(0, 3, n_evaluations)
    ridge_losses = []
    lasso_losses = []
    for lam in lams:
        ridge_model = RidgeRegression(lam)
        lasso_model = Lasso(lam)
        ridge_model.fit(X_train, y_train)
        lasso_model.fit(X_train, y_train)
        ridge_losses.append(
            cross_validate(ridge_model, X_train, y_train, mean_square_error))
        lasso_losses.append(
            cross_validate(lasso_model, X_train, y_train, mean_square_error))

    fig = go.Figure(go.Scatter(x=lams,
                               y=[i[0] for i in ridge_losses],
                               name="ridge train error"))
    fig.add_trace(go.Scatter(x=lams,
                             y=[i[1] for i in ridge_losses],
                             name="ridge validation error"
                             ))
    fig.add_trace(go.Scatter(x=lams,
                             y=[i[0] for i in lasso_losses],
                             name="lasso train error"
                             ))
    fig.add_trace(go.Scatter(x=lams,
                             y=[i[1] for i in lasso_losses],
                             name="lasso validation error"
                             ))
    fig.show()
    r_val = [i[1] for i in ridge_losses]
    l_val = [i[1] for i in lasso_losses]

    best_lam_ridge = lams[np.argmin(r_val)]
    best_lam_lasso = lams[np.argmin(l_val)]

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_model_best = RidgeRegression(best_lam_ridge)
    lasso_model_best = Lasso(best_lam_lasso)
    ls_model = LinearRegression()

    ridge_model_best.fit(X_train, y_train)
    lasso_model_best.fit(X_train, y_train)
    ls_model.fit(X_train, y_train)

    ridge_pred = ridge_model_best.predict(X_test)
    lasso_pred = lasso_model_best.predict(X_test)
    ls_pred = ls_model.predict(X_test)

    ridge_loss = mean_square_error(ridge_pred, y_test)
    lasso_loss = mean_square_error(lasso_pred, y_test)
    ls_pred = mean_square_error(ls_pred, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
