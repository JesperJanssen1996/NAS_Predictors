from BasePredictor import BasePredictor
from pySOT.surrogate import RBFInterpolant, CubicKernel, TPSKernel, LinearTail, ConstantTail
from sklearn.base import BaseEstimator, RegressorMixin
from RBF import RBF
from sklearn.model_selection import GridSearchCV

class RBFPredictor(BasePredictor):
    def __init__(self, params):
        super().__init__('RBF')
        self.params = params
        self.model = None

    def grid_search(self, x_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error'):
        # Perform grid search to find the best parameters
        grid_search = GridSearchCV(
            estimator=RBF(),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        grid_search.fit(x_train, y_train)
        self.params = grid_search.best_params_  # Save the best parameters
        return self.params

    def pretrain(self, x_pretrain, y_pretrain):
        # Pretrain using parameters from the grid search
        self.model = RBF()
        self.model.fit(x_pretrain, y_pretrain)

    def train(self, x_train, y_train):
        if not self.model:
            if self.params:
                self.model = RBF(**self.params)
            else:
                self.model = RBF()
        # self._inspect_parameters("Before Training")
        self.model.fit(x_train, y_train)
        # Force fitting by predicting on training data
        # _ = self.model.predict(x_train[:1])
        # self._inspect_parameters("After Training")

        
    # def _inspect_parameters(self, stage):
    #     """Inspect and print model parameters."""
    #     if self.model is None or not hasattr(self.model, 'get_learned_params'):
    #         print(f"{stage}: Model is not initialized or lacks learned parameters.")
    #     else:
    #         params = self.model.get_learned_params()
    #         if params is None or params['coefficients'] is None:
    #             print(f"{stage}: Learned parameters are not available.")
    #         else:
    #             print(f"{stage}: Learned parameters:")
    #             print(f"  Coefficients: {params['coefficients']}")
    #             print(f"  Nodes shape: {params['nodes'].shape if params['nodes'] is not None else None}")


    def predict(self, x_test):
        # Make predictions
        return self.model.predict(x_test)
