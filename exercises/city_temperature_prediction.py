import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"])
    full_data.dropna(inplace=True)

    full_data.drop(full_data[(full_data["Year"] <= 0) |
                             (full_data["Month"] <= 0) |
                             (full_data["Day"] <= 0)].index,
                   inplace=True)

    full_data.drop(full_data[(full_data["Temp"] < -70)].index, inplace=True)

    full_data["day_of_year"] = full_data["Date"].dt.dayofyear

    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    full_data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = full_data[full_data["Country"] == "Israel"]
    discrete_years = israel_data["Year"].astype(str)
    fig = px.scatter(israel_data, x='day_of_year', y='Temp',
                     color=discrete_years)
    fig.update_layout(
        title="Temperature in Israel as a function of the day of year")
    fig.show()

    israel_temp_by_month = \
        israel_data.groupby("Month")["Temp"].agg("std").reset_index()
    fig1 = px.bar(israel_temp_by_month, x="Month", y='Temp')
    fig1.update_layout(
        title="Std of Temperature in Israel as a Function of the Month",
        yaxis_title="std of the temperature")
    fig1.show()

    # Question 3 - Exploring differences between countries
    temp_by_country_month = full_data.groupby(["Country", "Month"])\
        .Temp.agg(["mean", "std"]).reset_index()
    fig2 = px.line(temp_by_country_month, x="Month", y="mean", error_y="std",
                   color="Country",
                   title="Average temperature in each country by month")
    fig2.show()

    # Question 4 - Fitting model for different values of `k`
    X = israel_data["day_of_year"]
    y = israel_data["Temp"]
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        losses.append(round(model.loss(test_X.to_numpy(), test_y.to_numpy()),
                           2))
    fig3 = px.bar(losses, x=range(1, 11), y=losses)
    fig3.update_layout(title="Loss as a function of the polynomial degree",
                      xaxis_title="degree",
                      yaxis_title="loss")
    fig3.show()

    # Question 5 - Evaluating fitted model on different countries
    isr_model = PolynomialFitting(3)
    isr_model.fit(X, y)
    countries = list(full_data["Country"].unique())
    countries.remove("Israel")
    losses = []
    for country in countries:
        data = full_data[full_data["Country"] == country]
        X = data["day_of_year"]
        y = data["Temp"]
        loss = isr_model.loss(X, y)
        losses.append([country, loss])
    df = pd.DataFrame(losses, columns=["Country", "loss"])
    fig4 = px.bar(df, x="Country", y="loss",
                  title="Prediction error for all countries when "
                        "prediction made with israel data fitted model")

    fig4.show()






