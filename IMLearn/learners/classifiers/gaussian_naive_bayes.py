from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.mu_ = np.empty((self.classes_.shape[0], X.shape[1]))
        self.vars_ = np.empty((self.classes_.shape[0], X.shape[1]))
        self.pi_ = np.empty((self.classes_.shape[0],))

        for idx, clss in enumerate(self.classes_):
            bool_array = np.where(y == clss, True, False)
            filtered_x = X[bool_array]
            self.mu_[idx] = filtered_x.mean(axis=0)
            self.vars_[idx] = filtered_x.var(axis=0, ddof=1)
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
        likelihood = self.likelihood(X)
        predictions = np.argmax(likelihood, axis=1)
        predictions = self.classes_[predictions]
        return predictions

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

        likelihood_mat = np.empty((X.shape[0], self.classes_.shape[0]))
        for i, x in enumerate(X):
            for k, pi_k in enumerate(self.pi_):
                log_pi_k = np.log(pi_k)
                log_likelihood = 0
                for d in range(X.shape[1]):
                    a = 1 / np.sqrt(2 * np.pi * self.vars_[k][d])
                    b = np.square(x[d] - self.mu_[k][d]) / (2 * self.vars_[k][d])
                    log_likelihood += np.log(a) - (b/2)

                likelihood_mat[i][k] = log_pi_k + log_likelihood

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
