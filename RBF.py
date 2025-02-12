from pySOT.surrogate import RBFInterpolant, CubicKernel, TPSKernel, LinearTail, ConstantTail, LinearKernel
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class RBF(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='linear', tail='constant'):
        self.kernel = kernel
        self.tail = tail
        self.model = None

    def fit(self, X, y):
        #print("Starting fit method...")
        # Choose kernel
        if self.kernel == 'cubic':
            kernel = CubicKernel
        elif self.kernel == 'tps':
            kernel = TPSKernel
        elif self.kernel == 'linear':
            kernel = LinearKernel
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Choose tail
        if self.tail == 'linear':
            tail = LinearTail
        elif self.tail == 'constant':
            tail = ConstantTail

        else:
            raise ValueError(f"Unknown tail: {self.tail}")
        
        lb = X.min(axis=0)  # Lower bounds
        ub = X.max(axis=0)  # Upper bounds

        # Initialize the RBF model
        self.model = RBFInterpolant(
            dim=X.shape[1], kernel=kernel(), tail=tail(X.shape[1]), lb=lb, ub=ub
        )

        # Add training points
        #print("Adding training points...")
        y = np.array(y)
        for i in range(len(X)):
            self.model.add_points(X[i, :], y[i])

        # Fit the model immediately
        #print("Fitting the model...")
        try:
            self.model._fit()
         #   print("Model fitting completed.")
        except Exception as e:
            print("Error during fitting:", e)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("RBF model is not trained yet. Call fit first.")
        return np.array([self.model.predict(x) for x in np.array(X)]).flatten()

    def get_learned_params(self):
        if self.model is not None:
            # Print available attributes
            print("Available attributes in self.model:", dir(self.model))
            
            # Get coefficients
            if hasattr(self.model, 'c'):
                coefficients = getattr(self.model, 'c')
            elif hasattr(self.model, '_c'):
                coefficients = getattr(self.model, '_c')
            else:
                coefficients = None
    
            # Get nodes
            if hasattr(self.model, 'X'):
                nodes = getattr(self.model, 'X')
            elif hasattr(self.model, '_X'):
                nodes = getattr(self.model, '_X')
            else:
                nodes = None
    
            return {
                'coefficients': coefficients,
                'nodes': nodes
            }
        else:
            return None
