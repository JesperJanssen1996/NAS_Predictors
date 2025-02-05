from BasePredictor import BasePredictor
from sklearn.base import BaseEstimator, RegressorMixin
from MLP import MLP
from itertools import product
from validate import validate
from sklearn.model_selection import KFold
import numpy as np


class MLPPredictor(BasePredictor):
    def __init__(self, params):
        super().__init__('MLP')
        self.params = params
        self.model = None

    def grid_search(self, x_train, y_train, param_grid, cv=2, best_params = None, scoring='neg_mean_squared_error'):
        if best_params != None:
            self.params = best_params
        else:
            param_combinations = list(product(*param_grid.values()))
            best_loss = float('inf')
            best_params = None
        
            # Create K-Folds for cross-validation
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
            # Evaluate each parameter combination
            for params in param_combinations:
                print(params)
                param_dict = dict(zip(param_grid.keys(), params))
                fold_losses = []
        
                for train_idx, val_idx in kf.split(x_train):
                    print(train_idx.shape)
                    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
                    # Create an MLP instance with the given parameters
                    model = MLP(
                        n_feature=x_train.shape[1],
                        n_layers=param_dict['n_layers'],
                        n_hidden=param_dict['n_hidden'],
                        drop=param_dict['drop']
                    )
        
                    # Train the model on the training fold
                    model.fit(
                        x=x_train_fold, 
                        y=y_train_fold, 
                        x_val=x_val_fold, 
                        y_val=y_val_fold, 
                        lr=param_dict['lr']
                    )
        
                    # Validate the model on the validation fold
                    rmse, _, _, _, _ = validate(model.model, x_val_fold, y_val_fold, device='cpu')
                    print(rmse)
                    fold_losses.append(rmse)
        
                # Compute the average RMSE across folds
                avg_loss = np.mean(fold_losses)
        
                # Update the best params if this is the best average loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.params = param_dict
    
        return self.params




    def pretrain(self, x_pretrain, y_pretrain):
        # Pretrain using parameters from the grid search
        self.model = MLP(n_feature=x_pretrain.shape[1],
                         **dict(list(self.params.items())[:3]))
        self.model.fit(x_pretrain, y_pretrain)

    def train(self, x_train, y_train):
        # Train using parameters from the grid search
        if not self.model:
            self.model = MLP(n_feature=x_train.shape[1],
                n_layers=self.params['n_layers'],
                n_hidden=self.params['n_hidden'],
                drop=self.params['drop']
                )
        self.model.fit(x_train, y_train, lr=self.params['lr'])

    def predict(self, x_test):
        # Make predictions
        return self.model.predict(x_test)
