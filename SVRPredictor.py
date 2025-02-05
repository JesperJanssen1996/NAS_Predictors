from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR

class SVRPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        """
        Wrapper around sklearn's SVR to handle custom parameter passing
        (no pretraining logic typically).
        """
        self.params = kwargs
        self.model = None

    def fit(self, X, y):
        self.model = SVR(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        # Return all hyperparams
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self
