from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.mu_ = np.empty((self.classes_.shape[0], X.shape[1]))
        self.pi_ = np.empty((self.classes_.shape[0],))

        for idx, clss in enumerate(self.classes_):
            bool_array = np.where(y == clss, True, False)
            filtered_x = X[bool_array]
            self.mu_[idx] = filtered_x.mean(axis=0)
            if self.cov_ is None:
                self.cov_ = np.cov(filtered_x.T, )
                self._cov_inv = inv(self.cov_)
            self.pi_[idx] = counts[idx] / X.shape[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        def calc_y_hat(k):
            a_k = self._cov_inv @ self.mu_[k]
            b_k = np.log(self.pi_[k]) - (
                    (1 / 2) * (self.mu_[k] @ self._cov_inv @ self.mu_[k]))
            return (a_k.T @ x) + b_k

        labels = []
        for x in X:
            probs = []
            for k in range(np.size(self.classes_)):
                probs.append((calc_y_hat(k)))
            labels.append(self.classes_[np.argmax(probs)])
        return np.asarray(labels)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        d = len(X[0])
        pi2_d = np.power(2 * np.pi, d)
        likelihood_mat = np.empty((X.shape[0], self.classes_.shape[0]))
        for i, x in enumerate(X):
            for k, pi_k in enumerate(self.pi_):
                x_muk = x - self.mu_[k]
                likelihood = (1 / (np.sqrt(pi2_d * det(self.cov_)))) * (
                    np.exp((-1 / 2) * (x_muk.T @ self._cov_inv @ x_muk))
                    ) * self.pi_[k]
                likelihood_mat[i][k] = likelihood

        return likelihood_mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_hat = self.predict(X)
        return misclassification_error(y, y_hat)
