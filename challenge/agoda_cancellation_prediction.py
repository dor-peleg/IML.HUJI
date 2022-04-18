from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename)
    features = full_data[:]
    labels = full_data["cancellation_datetime"]
    full_data.head()

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    df = pd.get_dummies(df, columns=['hotel_id',
                                     'hotel_country_code',
                                     'hotel_star_rating',
                                     'accommadation_type_name',
                                     'charge_option',
                                     'origin_country_code',
                                     'original_payment_type',
                                     'original_payment_currency',
                                     'hotel_area_code',
                                     'hotel_city_code'])
    cancellation_labels.head()
    # train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    #
    # # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    #
    # # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
