from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class XGBOOSTPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, pretraining=False, booster_state=None, **kwargs):
        """
        Wrapper around XGBRegressor to handle potential pretraining logic
        and custom parameter passing.
        """
        super().__init__()
        self.params = kwargs
        self.model = None
        self.booster_state = booster_state
        self.pretraining = pretraining

    def pretrain(self, model, X_pretrain, y_pretrain):
        """
        Train on a separate pretraining dataset, store the booster state.
        """
        self.model = model
        self.model.fit(X_pretrain, y_pretrain, eval_set=[(X_pretrain, y_pretrain)], verbose=False)
        self.booster_state = self.model.get_booster().save_raw()  # Save booster state
        return self.model

    def fit(self, X, y):
        """
        Fit XGBoost on the provided data. If we already have a booster_state,
        load it for fine-tuning; otherwise, train from scratch.
        """
        if self.booster_state and self.booster_state != "NO_PRETRAINING":
            self.model = XGBRegressor(**self.params, objective='reg:squarederror')
            self.model.load_model(self.booster_state)
            self.model.fit(X, y, eval_set=[(X, y)], verbose=False, xgb_model=self.model.get_booster())
        else:
            self.model = XGBRegressor(**self.params, objective='reg:squarederror')
            self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Return hyperparams for scikit-learn compatibility.
        The 'params' dict holds XGB-specific parameters.
        """
        out = dict(self.params)
        out['booster_state'] = self.booster_state
        out['pretraining'] = self.pretraining
        return out

    def set_params(self, **params):
        """
        Update hyperparams or booster_state.
        """
        self.booster_state = params.pop('booster_state', self.booster_state)
        self.pretraining = params.pop('pretraining', self.pretraining)
        self.params.update(params)
        return self
