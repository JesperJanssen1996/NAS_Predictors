import numpy as np
import pandas as pd
import time
import logging
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import ast
from calculating_FLOPs_and_params import calculate_FLOPs_and_Params

# Custom predictors
from XGBOOSTPredictor import XGBOOSTPredictor
from RBFPredictor import RBFPredictor
from SVRPredictor import SVRPredictor
from MLPPredictor import MLPPredictor

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from remove_duplicates_and_adjust_targets import remove_duplicates_and_adjust_targets
from clear_prefixes import clear_prefixes


def c_index_scorer(y_true, y_pred):
    return concordance_index(y_true, y_pred)


class PredictorPipeline:
    def __init__(
        self,
        data_path,
        flop_path,
        param_path,
        N_layers_path,
        pretrain_pct,
        pretraining,
        preprocessing,
        remove_duplicates=False,
        flops_calc=False,
        pretrain_label='N_layers',
        train_pct=False
    ):
        """
        Initialize the pipeline with data usage arguments and pretraining details.

        :param data_path: Path to the CSV file.
        :param flop_path: Path to the FLOPs .csv file (for pretraining if needed).
        :param param_path: Path to the Params .csv file (for pretraining if needed).
        :param N_layers_path: Path to the N_layers .csv file (for pretraining if needed).
        :param pretrain_pct: Fraction of data for pretraining (0.0 => no pretraining data).
        :param pretraining: Boolean - whether to use pretraining or not.
        :param preprocessing: Boolean - whether to apply preprocessing (scaling, encoding).
        :param remove_duplicates: Boolean - if duplicates should be removed (not used here).
        :param flops_calc: Boolean - whether to calculate FLOPs (not fully shown here).
        :param pretrain_label: Which label to use for pretraining ('FLOPs', 'Params', or 'N_layers').
        :param train_pct: If True, sub-sample the training set (not used in scenario 3).
        """
        self.data_path = data_path
        self.flop_path = flop_path
        self.param_path = param_path
        self.N_layers_path = N_layers_path
        self.pretrain_pct = pretrain_pct
        self.pretraining = pretraining
        self.preprocessing = preprocessing
        self.remove_duplicates = remove_duplicates
        self.flops_calc = flops_calc
        self.pretrain_label = pretrain_label
        self.train_pct = train_pct

        # Load data
        self.data = pd.read_csv(self.data_path, header=None)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data_numeric = self.data[0].apply(ast.literal_eval)

    def prepare_data(self, model):
        """
        Prepare data for training, applying slicing for pretraining if needed,
        and setting up the ColumnTransformer if self.preprocessing=True.
        """
        # Define column indices
        self.numerical_indices = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26]
        self.categorical_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25]
        all_indices = list(range(28))
        self.ordinal_indices = sorted(list(set(all_indices) - set(self.numerical_indices + self.categorical_indices)))

        # Choose whether to drop='first' in OneHotEncoder for certain models
        if model in ['RBF', 'MLP', 'SVR']:
            cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        else:
            cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Build the ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_indices),
                ('cat', cat_transformer, self.categorical_indices),
                ('ord', StandardScaler(), self.ordinal_indices)
            ]
        )

        # Convert the data to numpy arrays
        X = np.array([eval(row[0]) for row in self.data.values])
        y = self.data.iloc[:, -1].values

        # Pretrain slicing
        n_pretrain = int(self.pretrain_pct * y.shape[0])
        # For scenario 3, pretrain_pct=0 => no slicing
        if self.train_pct:
            # If you were sub-sampling training data, handle that logic here
            pass

        # The main (post-pretrain) training set
        self.X = X[n_pretrain:]
        self.Y = y[n_pretrain:]

    # ------------------------------------------
    # XGBoost Pipeline
    # ------------------------------------------
    def run_XGBoostPipeline(self, param_grid, model='XGB'):
        """Run XGBoost with grid search and cross_val_predict."""
        self.prepare_data(model)
        xgb_predictor = XGBOOSTPredictor()

        c_index = make_scorer(c_index_scorer, greater_is_better=True)

        if self.preprocessing:
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('xgb', xgb_predictor)
            ])
        else:
            pipeline = Pipeline([
                ('xgb', xgb_predictor)
            ])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=c_index, cv=5, n_jobs=-1)
        grid_search.fit(self.X, self.Y)

        print("Global best params:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        xgb_predictions = cross_val_predict(estimator=best_model, X=self.X, y=self.Y, cv=10)
        xgb_r2 = r2_score(self.Y, xgb_predictions)
        xgb_mse = mean_squared_error(self.Y, xgb_predictions)
        xgb_c_index = concordance_index(self.Y, xgb_predictions)

        # Return performance metrics
        return round(xgb_r2, 4), round(xgb_mse, 4), grid_search.best_params_, round(xgb_c_index, 4)

    # ------------------------------------------
    # SVR Pipeline
    # ------------------------------------------
    def run_SVRPipeline(self, param_grid, model='SVR'):
        """Run SVR with grid search and cross_val_predict."""
        self.prepare_data(model)
        svr_predictor = SVRPredictor()

        c_index = make_scorer(c_index_scorer, greater_is_better=True)

        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('svr', svr_predictor)
        ]) if self.preprocessing else Pipeline([('svr', svr_predictor)])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=c_index, cv=5, n_jobs=-1)
        grid_search.fit(self.X, self.Y)
        print("Global best params:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        svr_predictions = cross_val_predict(estimator=best_model, X=self.X, y=self.Y, cv=10)
        svr_r2 = r2_score(self.Y, svr_predictions)
        svr_mse = mean_squared_error(self.Y, svr_predictions)
        svr_c_index = concordance_index(self.Y, svr_predictions)

        return round(svr_r2, 4), round(svr_mse, 4), grid_search.best_params_, round(svr_c_index, 4)

    # ------------------------------------------
    # RBF Pipeline
    # ------------------------------------------
    def run_RBFPipeline(self, param_grid, model='RBF'):
        """Run RBF with grid search and cross_val_predict."""
        self.prepare_data(model)
        rbf_predictor = RBFPredictor()

        c_index = make_scorer(c_index_scorer, greater_is_better=True)

        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('rbf', rbf_predictor)
        ]) if self.preprocessing else Pipeline([('rbf', rbf_predictor)])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=c_index, cv=5, n_jobs=-1)
        grid_search.fit(self.X, self.Y)
        print("Global best params:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        rbf_predictions = cross_val_predict(estimator=best_model, X=self.X, y=self.Y, cv=10)
        rbf_r2 = r2_score(self.Y, rbf_predictions)
        rbf_mse = mean_squared_error(self.Y, rbf_predictions)
        rbf_c_index = concordance_index(self.Y, rbf_predictions)

        return round(rbf_r2, 4), round(rbf_mse, 4), grid_search.best_params_, round(rbf_c_index, 4)

    # ------------------------------------------
    # MLP Pipeline
    # ------------------------------------------
    def run_MLPPipeline(self, param_grid, model='MLP'):
        """Run MLP with grid search and cross_val_predict."""
        self.prepare_data(model)
        mlp_predictor = MLPPredictor()
        c_index = make_scorer(c_index_scorer, greater_is_better=True)

        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('mlp', mlp_predictor)
        ]) if self.preprocessing else Pipeline([('mlp', mlp_predictor)])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=c_index, cv=5, n_jobs=-1)
        grid_search.fit(self.X, self.Y)
        print("Global best params:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        mlp_predictions = cross_val_predict(estimator=best_model, X=self.X, y=self.Y, cv=10)
        mlp_r2 = r2_score(self.Y, mlp_predictions)
        mlp_mse = mean_squared_error(self.Y, mlp_predictions)
        mlp_c_index = concordance_index(self.Y, mlp_predictions)

        return round(mlp_r2, 4), round(mlp_mse, 4), grid_search.best_params_, round(mlp_c_index, 4)


if __name__ == "__main__":
    # Define file paths
    data_path = 'C:/Users/jesper/.spyder-py3/model_psnr_result.csv'
    flop_path = 'C:/Users/jesper/.spyder-py3/FLOPs_pretrain.csv'
    param_path = 'C:/Users/jesper/.spyder-py3/Params_pretrain.csv'
    N_layers_path = 'C:/Users/jesper/.spyder-py3/N_layers_pretrain.csv'
    
    # Parameter grids
    xgb_param_grid = {
        'xgb__n_estimators': [250, 350, 400],
        'xgb__max_depth': [3, 5, 7, 9],
        'xgb__learning_rate': [0.03, 0.05, 0.07],
        'xgb__subsample': [0.6, 0.7, 0.8]
    }
    svr_param_grid = {
        'svr__C': [3, 5, 8, 10],
        'svr__gamma': ['scale', 'auto'],
        'svr__kernel': ['linear', 'rbf', 'poly']
    }
    rbf_param_grid = {
        'rbf__n_centers': [30, 60, 80, 100],
        'rbf__gamma': [0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.01, 0.05, 0.1, 0.5],
        'rbf__alpha': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        'rbf__center_selection': ['kmeans'],
        'rbf__kernel': ['gaussian', 'multiquadric', 'inverse_multiquadric']
    }
    mlp_param_grid = {
        'mlp__n_layers': [2, 4, 5],
        'mlp__n_hidden': [64, 128, 256, 512],
        'mlp__drop': [0, 0.05, 0.1, 0.2, 0.4],
        'mlp__lr': [0.0005, 0.001, 0.005, 0.01]
    }

    # Loops for scenarios
    pretrain_pct = [0.1]
    preprocessing_options = [True]
    pretraining_options = [True, False]  # if you want to add them for other scenarios

    # DataFrames for final results
    results_with_pretraining = []
    results_no_pretraining = []
    results_full_no_pretraining = []

    # Run the main loops
    for preprocessing in preprocessing_options:
        for pct in pretrain_pct:
            if pct > 0.0:
                # You can handle partial data with or without pretraining if desired
                for prtrn in pretraining_options:
                    # Initialize pipeline
                    pipeline = PredictorPipeline(
                        data_path=data_path,
                        flop_path=flop_path,
                        param_path=param_path,
                        N_layers_path=N_layers_path,
                        pretrain_pct=pct,
                        pretraining=prtrn,
                        preprocessing=preprocessing
                    )
                    if prtrn:
                        # SCENARIO 1: partial data with pretraining
                
                        # XGBoost
                        xgb_r2, xgb_mse, xgb_best_params, xgb_c_index = pipeline.run_XGBoostPipeline(
                            param_grid=xgb_param_grid, model='XGB')
                        results_full_no_pretraining.append({
                            "Model": "XGBoost",
                            "R^2": xgb_r2,
                            "MSE": xgb_mse,
                            "C-Index": xgb_c_index,
                            "Preprocessing": preprocessing
                        })

                        # SVR
                        svr_r2, svr_mse, svr_best_params, svr_c_index = pipeline.run_SVRPipeline(
                            param_grid=svr_param_grid, model='SVR')
                        results_full_no_pretraining.append({
                            "Model": "SVR",
                            "R^2": svr_r2,
                            "MSE": svr_mse,
                            "C-Index": svr_c_index,
                            "Preprocessing": preprocessing
                        })

                        # RBF
                        rbf_r2, rbf_mse, rbf_best_params, rbf_c_index = pipeline.run_RBFPipeline(
                            param_grid=rbf_param_grid, model='RBF')
                        results_full_no_pretraining.append({
                            "Model": "RBF",
                            "R^2": rbf_r2,
                            "MSE": rbf_mse,
                            "C-Index": rbf_c_index,
                            "Preprocessing": preprocessing
                        })

                        # MLP
                        mlp_r2, mlp_mse, mlp_best_params, mlp_c_index = pipeline.run_MLPPipeline(
                            param_grid=mlp_param_grid, model='MLP')
                        results_full_no_pretraining.append({
                            "Model": "MLP",
                            "R^2": mlp_r2,
                            "MSE": mlp_mse,
                            "C-Index": mlp_c_index,
                            "Preprocessing": preprocessing
                        })

                    else:

                        # XGBoost
                        xgb_r2, xgb_mse, xgb_best_params, xgb_c_index = pipeline.run_XGBoostPipeline(
                            param_grid=xgb_param_grid, model='XGB')
                        results_full_no_pretraining.append({
                            "Model": "XGBoost",
                            "R^2": xgb_r2,
                            "MSE": xgb_mse,
                            "C-Index": xgb_c_index,
                            "Preprocessing": preprocessing
                        })

                        # SVR
                        svr_r2, svr_mse, svr_best_params, svr_c_index = pipeline.run_SVRPipeline(
                            param_grid=svr_param_grid, model='SVR')
                        results_full_no_pretraining.append({
                            "Model": "SVR",
                            "R^2": svr_r2,
                            "MSE": svr_mse,
                            "C-Index": svr_c_index,
                            "Preprocessing": preprocessing
                        })

                        # RBF
                        rbf_r2, rbf_mse, rbf_best_params, rbf_c_index = pipeline.run_RBFPipeline(
                            param_grid=rbf_param_grid, model='RBF')
                        results_full_no_pretraining.append({
                            "Model": "RBF",
                            "R^2": rbf_r2,
                            "MSE": rbf_mse,
                            "C-Index": rbf_c_index,
                            "Preprocessing": preprocessing
                        })

                        # MLP
                        mlp_r2, mlp_mse, mlp_best_params, mlp_c_index = pipeline.run_MLPPipeline(
                            param_grid=mlp_param_grid, model='MLP')
                        results_full_no_pretraining.append({
                            "Model": "MLP",
                            "R^2": mlp_r2,
                            "MSE": mlp_mse,
                            "C-Index": mlp_c_index,
                            "Preprocessing": preprocessing
                        })
            else:
                # SCENARIO 3: Full Data Set, No Pretraining
                logging.info("Running Scenario: Full Data, No Pretraining")

                pipeline = PredictorPipeline(
                    data_path=data_path,
                    flop_path=flop_path,
                    param_path=param_path,
                    N_layers_path=N_layers_path,
                    pretrain_pct=0.0,  # full data
                    pretraining=False,
                    preprocessing=preprocessing
                )

                # XGBoost
                xgb_r2, xgb_mse, xgb_best_params, xgb_c_index = pipeline.run_XGBoostPipeline(
                    param_grid=xgb_param_grid, model='XGB')
                results_full_no_pretraining.append({
                    "Model": "XGBoost",
                    "R^2": xgb_r2,
                    "MSE": xgb_mse,
                    "C-Index": xgb_c_index,
                    "Preprocessing": preprocessing
                })

                # SVR
                svr_r2, svr_mse, svr_best_params, svr_c_index = pipeline.run_SVRPipeline(
                    param_grid=svr_param_grid, model='SVR')
                results_full_no_pretraining.append({
                    "Model": "SVR",
                    "R^2": svr_r2,
                    "MSE": svr_mse,
                    "C-Index": svr_c_index,
                    "Preprocessing": preprocessing
                })

                # RBF
                rbf_r2, rbf_mse, rbf_best_params, rbf_c_index = pipeline.run_RBFPipeline(
                    param_grid=rbf_param_grid, model='RBF')
                results_full_no_pretraining.append({
                    "Model": "RBF",
                    "R^2": rbf_r2,
                    "MSE": rbf_mse,
                    "C-Index": rbf_c_index,
                    "Preprocessing": preprocessing
                })

                # MLP
                mlp_r2, mlp_mse, mlp_best_params, mlp_c_index = pipeline.run_MLPPipeline(
                    param_grid=mlp_param_grid, model='MLP')
                results_full_no_pretraining.append({
                    "Model": "MLP",
                    "R^2": mlp_r2,
                    "MSE": mlp_mse,
                    "C-Index": mlp_c_index,
                    "Preprocessing": preprocessing
                })

    # Save final results for scenario 3 (full data, no pretraining)
    # Distinguish with vs. without preprocessing
    df_full_no_pretraining = pd.DataFrame(results_full_no_pretraining)

    # Separate them by Preprocessing
    df_with_preprocessing = df_full_no_pretraining[df_full_no_pretraining["Preprocessing"] == True]
    df_without_preprocessing = df_full_no_pretraining[df_full_no_pretraining["Preprocessing"] == False]

    # Write to CSV
    df_with_preprocessing.to_csv("Results_Full_No_Pretraining_with_preprocessing.csv", index=False)
    df_without_preprocessing.to_csv("Results_Full_No_Pretraining_without_preprocessing.csv", index=False)

    # Optional prints
    print("\nResults Full No Pretraining - With Preprocessing:")
    print(df_with_preprocessing)

    print("\nResults Full No Pretraining - Without Preprocessing:")
    print(df_without_preprocessing)
