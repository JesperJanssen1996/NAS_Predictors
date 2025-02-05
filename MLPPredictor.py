from sklearn.base import BaseEstimator, RegressorMixin
import copy
import torch
import numpy as np
import torch.nn as nn
# Suppose you have an MLP class in an external file 'MLP.py'
from MLP import MLP
from sklearn.model_selection import GridSearchCV

class MLPPredictor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_layers=2,
                 n_hidden=64,
                 drop=0.1,
                 lr=1e-3,
                 epochs=3000,
                 model_state=None):
        """
        Example PyTorch MLP wrapper for scikit-learn.
        """
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.drop = drop
        self.lr = lr
        self.epochs = epochs
        self.model_state = model_state
        self.model = None

    def pretrain(self, X_pretrain, y_pretrain):
        """
        Pretrain on separate data, store weights in self.model_state
        """
        temp_model = MLP(
            n_feature=X_pretrain.shape[1],
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            drop=self.drop
        )
        temp_model.fit(X_pretrain, y_pretrain, device='cpu', lr=self.lr, epochs=self.epochs)
        self.model_state = copy.deepcopy(temp_model.model.state_dict())

    def fit(self, X, y):
        if self.model is None:
            # If we have a saved state, create an MLP and load it
            self.model = MLP(
                n_feature=X.shape[1],
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                drop=self.drop
            )
            if self.model_state is not None:
                self.model.model.load_state_dict(self.model_state)

        # Fine-tune on new data
        self.model.fit(X, y, device='cpu', lr=self.lr, epochs=self.epochs)
        self.model_state = copy.deepcopy(self.model.model.state_dict())
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("The MLP model has not been trained yet.")
        return self.model.predict(X, device='cpu').ravel()

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'n_hidden': self.n_hidden,
            'drop': self.drop,
            'lr': self.lr,
            'epochs': self.epochs,
            'model_state': self.model_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'model_state':
                self.model_state = value
            else:
                setattr(self, key, value)
        return self
