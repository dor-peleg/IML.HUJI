from IMLearn import metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        per = Perceptron()

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit, callback_x, callbak_y):
            callback_x = X
            callbak_y = y
            losses.append(fit.loss(callback_x, callbak_y))

        per.callback_ = callback
        per.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=np.arange(0, len(losses)), y=losses))
        fig.update_layout(
            title="Perceptron loss as function of fitting iteration\n "
                  f"data: {f}",
            xaxis_title="iteration number",
            yaxis_title="loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        pred_lda = lda.predict(X)
        pred_gnb = gnb.predict(X)

        lda_acc = accuracy(y, pred_lda)
        gnb_acc = accuracy(y, pred_gnb)

        symbols = np.array(["circle", "x", "square"])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian Naive Bayes Classifier "
                                f"<br>accuracy = {gnb_acc}",
                                f"LDA Classifier "
                                f"<br>accuracy = {lda_acc}"))
        fig.update_layout(title_text=f"Classifier Comparison <br>Data: {f}",
                          title_x=0.5, margin_t=100,
                          xaxis_title="First feature",
                          yaxis_title="Second feature")

        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=pred_lda,
                                   symbol=symbols[y])),
            row=1, col=2)

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=pred_gnb,
                                   symbol=symbols[y])),
            row=1, col=1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                             size=10)),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                             size=10)),
                      row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(lda.classes_.shape[0]):
            fig.add_trace(get_ellipse(lda.mu_[k], lda.cov_),
                          row=1, col=2)

        for k in range(gnb.classes_.shape[0]):
            fig.add_trace(get_ellipse(gnb.mu_[k], np.diag(gnb.vars_[k])),
                          row=1, col=1)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
