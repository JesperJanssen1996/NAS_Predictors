from BasePredictor import BasePredictor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

class XGBOOSTPredictor(BasePredictor):
    def __init__(self, params):
        super().__init__('XGBOOST')
        self.params = params
        self.model = None

    def grid_search(self, x_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error'):
        # Perform grid search to find the best parameters
        grid_search = GridSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror'),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        grid_search.fit(x_train, y_train)
        self.params = grid_search.best_params_  # Save the best parameters
        return self.params

    def pretrain(self, x_pretrain, y_pretrain):
        # Pretrain using parameters from the grid search
        self.model = XGBRegressor(
            **self.params,  # Use params directly from grid search
            objective='reg:squarederror'
        )
        self.model.fit(x_pretrain, y_pretrain, eval_set=[(x_pretrain, y_pretrain)], verbose=False)

    def train(self, x_train, y_train):
        # Train using parameters from the grid search
        if not self.model:
            self.model = XGBRegressor(
                **self.params,  # Use params directly from grid search
                objective='reg:squarederror'
            )
        self.model.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=False)

    def predict(self, x_test):
        # Make predictions
        return self.model.predict(x_test)
