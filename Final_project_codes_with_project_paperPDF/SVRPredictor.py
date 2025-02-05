from BasePredictor import BasePredictor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class SVRPredictor(BasePredictor):
    def __init__(self, params):
        super().__init__('SVR')
        self.params = params
        self.model = None

    def grid_search(self, x_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error'):
        # Perform grid search to find the best parameters
        grid_search = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        grid_search.fit(x_train, y_train)
        self.params = grid_search.best_params_  # Save the best parameters
        return self.params

    def pretrain(self, x_pretrain, y_pretrain):
        # Pretrain using parameters from the grid search
        self.model = SVR(
            **self.params,  # Use params directly from grid search
        )
        self.model.fit(x_pretrain, y_pretrain)

    def train(self, x_train, y_train):
        # Train using parameters from the grid search
        if not self.model:
            self.model = SVR(
                **self.params,  # Use params directly from grid search
            )
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        # Make predictions
        return self.model.predict(x_test)