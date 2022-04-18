from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename)
    full_data.dropna(inplace=True)

    full_data.drop(full_data[full_data["price"] <= 0].index, inplace=True)
    full_data.drop(full_data[full_data["sqft_lot15"] < 0].index, inplace=True)


    features = full_data[:]
    labels = full_data["price"]

    # features["good_cond"] = np.where(features["condition"] >= 3, 1, 0)
    features = pd.get_dummies(features, columns=["zipcode"])
    features.drop(columns=[
        "price",
        "id",
        "date",
        "sqft_lot",
        "sqft_lot15",
        "lat",
        "long"
    ], inplace=True)

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        cov = np.ma.cov(X[feature], y)[0, 1]
        feature_std, response_std = np.std(X[feature]), np.std(y)
        corr = cov / (feature_std * response_std)

        fig = go.Figure(go.Scatter(x=X[feature], y=y, mode='markers'))

        fig.update_layout(title=f"Connection between {feature} "
                                f"and house price. "
                                f"Correlation = {round(corr, 3)}",
                          xaxis_title=feature,
                          yaxis_title="price")
        fig.write_image(output_path + f"{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    model = LinearRegression()

    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data(
        "../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, "")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    losses_data = []
    percents = np.arange(10, 101, 1)
    for percent in percents:
        losses = []
        for _ in range(10):
            partial_train = train_X.sample(frac=(percent / 100))
            partial_respones = train_y.reindex_like(partial_train)
            model.fit(partial_train.to_numpy(), partial_respones.to_numpy())
            loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
            losses.append(loss)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        losses_data.append([percent, mean_loss, std_loss])
    df = pd.DataFrame(losses_data, columns=["percents", "mean", "std"])
    fig = go.Figure([go.Scatter(x=df["percents"],
                                y=df["mean"]),

                     go.Scatter(x=df["percents"],
                                y=df["mean"] - 2 * df["std"],
                                mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False
                                ),

                     go.Scatter(x=df["percents"],
                                y=df["mean"] + 2 * df["std"],
                                fill='tonexty',
                                mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False
                                )])
    fig.update_layout(title="Mean Loss as a function of training size",
                      xaxis_title="Percentage of all samples",
                      yaxis_title="mean loss")
    fig.show()
