from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
# Assume RBFNN is some custom RBF neural net class you wrote:
from RBFNN import RBFNN

class RBFPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, pretraining=False,
                 n_centers=50,
                 kernel='gaussian',
                 gamma=0.002,
                 alpha=0.001,
                 center_selection='kmeans',
                 random_state=None,
                 centers=None,
                 weights=None):
        """
        Custom RBF predictor with optional pretraining state.
        """
        self.pretraining = pretraining
        self.n_centers = n_centers
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.center_selection = center_selection
        self.random_state = random_state
        self.centers = centers
        self.weights = weights
        self.model = None

    def pretrain(self, X_pretrain, y_pretrain):
        """
        Pretrain on separate data, storing centers and weights in self.
        """
        temp_model = RBFNN(
            n_centers=self.n_centers,
            kernel=self.kernel,
            gamma=self.gamma,
            alpha=self.alpha,
            center_selection=self.center_selection,
            random_state=self.random_state,
        )
        temp_model.fit(X_pretrain, y_pretrain)
        self.centers = temp_model.centers_
        self.weights = temp_model.weights_

    def fit(self, X, y):
        print("DEBUG: Entered RBFPredictor.fit()")
        if self.centers is not None and self.weights is not None:
            # Start from pre-trained centers/weights
            self.model = RBFNN(
                n_centers=self.n_centers,
                kernel=self.kernel,
                gamma=self.gamma,
                alpha=self.alpha,
                center_selection=self.center_selection,
                random_state=self.random_state,
                centers_=self.centers,
                weights_=self.weights
            )
        else:
            # Training from scratch
            self.model = RBFNN(
                n_centers=self.n_centers,
                kernel=self.kernel,
                gamma=self.gamma,
                alpha=self.alpha,
                center_selection=self.center_selection,
                random_state=self.random_state
            )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'pretraining': self.pretraining,
            'n_centers': self.n_centers,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'center_selection': self.center_selection,
            'random_state': self.random_state,
            'centers': self.centers,
            'weights': self.weights
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
