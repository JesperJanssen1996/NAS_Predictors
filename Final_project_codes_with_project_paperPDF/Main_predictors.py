import numpy as np
import pandas as pd
import time
import logging

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import ast
from calculating_FLOPs_and_params import calculate_FLOPs_and_Params
from XGBOOSTPredictor import XGBOOSTPredictor
from RBFPredictor import RBFPredictor
from SVRPredictor import SVRPredictor
from MLPPredictor import MLPPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from remove_duplicates_and_adjust_targets import remove_duplicates_and_adjust_targets


class PredictorPipeline:
    def __init__(self, data_path, flop_path, param_path, pretrain_pct, pretraining, remove_duplicates):
        """
        Initialize the pipeline with data and percentage splits.

        :param data_path: Path to the CSV file.
        :param flop_path: Path to the .npy file for FLOPs.
        :param param_path: Path to the .npy file for Params.
        :param pretrain_pct: Percentage of data for pretraining.
        """
        self.data_path = data_path
        self.flop_path = flop_path
        self.param_path = param_path
        self.pretrain_pct = pretrain_pct
        self.pretraining = pretraining
        self.remove_duplicates = remove_duplicates



        ''''perm = torch.randperm(target.size(0)) IS THIS NEEDED FOR ALL MY PREDICTORS TO STANDARDIZE COMPARISON?'''
        # Load data
        self.data = pd.read_csv(self.data_path, header=None)
        self.data_numeric = self.data[0].apply(ast.literal_eval)
        
        self.Y_pretrain_data = np.load(self.flop_path)
        self.Y_pretrain_data = ((self.Y_pretrain_data - np.mean(self.Y_pretrain_data))/np.std(self.Y_pretrain_data))*(max(self.data.iloc[:, -1].values)-min(self.data.iloc[:, -1].values))+min(self.data.iloc[:, -1].values)

        # Define feature indices based on your description
        self.numerical_indices = [0]  # 'channels'
        self.categorical_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25]  # 2nd, 5th, 8th, etc.

        # Generate ordinal indices by excluding numerical and categorical indices
        all_indices = list(range(28))
        self.ordinal_indices = list(set(all_indices) - set(self.numerical_indices + self.categorical_indices))
        self.ordinal_indices.sort()

        # Create the ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_indices),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_indices),
                ('ord', StandardScaler(), self.ordinal_indices)
            ]
        )

    def prepare_data(self, model):
        """Prepare data for pretraining, training, and testing based on the model."""
        # Convert the data to numpy arrays
        X = np.array([eval(row[0]) for row in self.data.values])
        y = self.data.iloc[:, -1].values
        y_pre = self.Y_pretrain_data
        # Verify input array lengths
        assert len(X) == len(y) == len(y_pre), "X, y, and Y_pretrain must have the same number of samples."
    
        if self.remove_duplicates:
            # Remove duplicates and adjust targets
            X_unique, y_unique, Y_pretrain_unique = remove_duplicates_and_adjust_targets(X, y, y_pre)
            print(f"Dataset size after removing duplicates: {len(X_unique)} samples.")
        else:
            # Use original data without removing duplicates
            X_unique, y_unique, Y_pretrain_unique = X, y, y_pre
            print(f"Using original dataset without removing duplicates: {len(X_unique)} samples.")
    
       
        # Determine the number of pretraining samples
        n_samples = len(X_unique)
        n_pretrain = int(self.pretrain_pct * n_samples)
         
        # Split data into pretraining and remaining data
        self.X_pretrain = X_unique[:n_pretrain]
        self.Y_pretrain = Y_pretrain_unique[:n_pretrain]
         
        X_remaining = X_unique[n_pretrain:]
        y_remaining = y_unique[n_pretrain:]
         
        # Split remaining data into training and testing sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.036, random_state=1)


        # #if model != 'RBF' and model != 'MLP':
        # # Fit the preprocessor on the training data
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        if self.pretrain_pct != 0.0:
             self.X_pretrain = self.preprocessor.transform(self.X_pretrain)
        self.X_test = self.preprocessor.transform(self.X_test)
        # #else:
        # # For RBF, no preprocessing is applied
        # # pass

    def run_XGBoostPipeline(self, param_grid, model='XGB'):
        """Run the XGBoost pipeline with grid search, pretraining, and evaluation."""
        # Prepare data
        self.prepare_data(model)

        # Grid Search
        xgb_predictor = XGBOOSTPredictor(params=None)
        xgb_best_params = xgb_predictor.grid_search(self.X_train, self.Y_train, param_grid)
        print("Best Parameters Found:", xgb_best_params)

        # Pretraining
        if self.pretrain_pct == 0.0:
            print('No pretraining')
        elif self.pretraining == True:
            print('with pretraining')
            xgb_predictor.pretrain(self.X_pretrain, self.Y_pretrain)
        else:
            print('same amount of training data no pretraining')
            
        # Training
        xgb_predictor.train(self.X_train, self.Y_train)

        # Prediction
        xgb_predictions = xgb_predictor.predict(self.X_test)

        # Metrics
        xgb_r2 = r2_score(self.Y_test, xgb_predictions)
        xgb_mse = mean_squared_error(self.Y_test, xgb_predictions)

        # Plot
        plt.scatter(self.Y_test, xgb_predictions)
        # Calculate the minimum and maximum values from both true and predicted values
        min_val = min(np.min(self.Y_test), np.min(xgb_predictions))
        max_val = max(np.max(self.Y_test), np.max(xgb_predictions))
        
        # Plot the Perfect Prediction Line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel("True Values (y_test)")
        plt.ylabel("Predicted Values (xgb_predictions)")
        plt.title("True vs Predicted Values")
        plt.savefig('NewNOn'f"True_vs_Predicted_{model}.png")
        plt.close()

        # Print Results
        print('For the XGB predictor the R^2 =', xgb_r2, '\nThe MSE =', xgb_mse)

        feature_importance = xgb_predictor.model.feature_importances_
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.show()
        return xgb_r2, xgb_mse, xgb_best_params

    def run_SVRPipeline(self, param_grid, model='SVR'):
        """Run the SVR pipeline with grid search, pretraining, and evaluation."""
        # Prepare data
        self.prepare_data(model)

        # Grid Search
        svr_predictor = SVRPredictor(params=None)
        svr_best_params = svr_predictor.grid_search(self.X_train, self.Y_train, param_grid)
        print("Best Parameters Found:", svr_best_params)

        # Pretraining
        if self.pretrain_pct == 0.0:
            print('No pretraining')
        elif self.pretraining == True:
            print('with pretraining')
            svr_predictor.pretrain(self.X_pretrain, self.Y_pretrain)
        else:
            print('same amount of training data no pretraining')
            
        # Training
        svr_predictor.train(self.X_train, self.Y_train)

        # Prediction
        svr_predictions = svr_predictor.predict(self.X_test)

        # Metrics
        svr_r2 = r2_score(self.Y_test, svr_predictions)
        svr_mse = mean_squared_error(self.Y_test, svr_predictions)

        # Plot
        plt.scatter(self.Y_test, svr_predictions)
        # Calculate the minimum and maximum values from both true and predicted values
        min_val = min(np.min(self.Y_test), np.min(svr_predictions))
        max_val = max(np.max(self.Y_test), np.max(svr_predictions))
        
        # Plot the Perfect Prediction Line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel("True Values (y_test)")
        plt.ylabel("Predicted Values (svr_predictions)")
        plt.title("True vs Predicted Values")
        plt.savefig('NewNOn'f"True_vs_Predicted_{model}.png")
        plt.close()

        # Print Results
        print('For the SVR predictor the R^2 =', svr_r2, '\nThe MSE =', svr_mse)
        return svr_r2, svr_mse, svr_best_params


    def run_RBFPipeline(self, param_grid, model='RBF'):
        """Run the RBF pipeline with pretraining and evaluation."""
        # Prepare data
        self.prepare_data(model)

        # Create the RBF predictor
        rbf_predictor = RBFPredictor(params=None)
        #best_params = rbf_predictor.grid_search(self.X_train, self.Y_train, param_grid)
        rbf_best_params = {'kernel': 'linear', 'tail':'constant'}
        #print('The best parameters are:',best_params)

        # Pretraining
        if self.pretrain_pct == 0.0:
            print('No pretraining')
        elif self.pretraining == True:
            print('with pretraining')
            rbf_predictor.pretrain(self.X_pretrain, self.Y_pretrain)
        else:
            print('same amount of training data no pretraining')            

        # Training
        rbf_predictor.train(self.X_train, self.Y_train)

        # Prediction
        rbf_predictions = rbf_predictor.predict(self.X_test)

        # Metrics
        rbf_r2 = r2_score(self.Y_test, rbf_predictions)
        rbf_mse = mean_squared_error(self.Y_test, rbf_predictions)

        # Plot
        plt.scatter(self.Y_test, rbf_predictions)
        # Calculate the minimum and maximum values from both true and predicted values
        min_val = min(np.min(self.Y_test), np.min(rbf_predictions))
        max_val = max(np.max(self.Y_test), np.max(rbf_predictions))
        
        # Plot the Perfect Prediction Line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel("True Values (y_test)")
        plt.ylabel("Predicted Values (rbf_predictions)")
        plt.title("True vs Predicted Values")
        plt.savefig('NewNOn'f"True_vs_Predicted_{model}.png")
        plt.close()

        # Print Results
        print('For the RBF predictor the R^2 =', rbf_r2, '\nThe MSE =', rbf_mse)
        return rbf_r2, rbf_mse, rbf_best_params


    def run_MLPPipeline(self, param_grid, model='MLP'):
        """Run the RBF pipeline with pretraining and evaluation."""
        # Prepare data
        self.prepare_data(model)

        # Create the RBF predictor
        mlp_predictor = MLPPredictor(params=None)
        mlp_best_params = mlp_predictor.grid_search(self.X_train, self.Y_train, param_grid, best_params = {'n_layers': 3, 'n_hidden': 500, 'drop': 0.2, 'lr': 0.005})
        print('The best parameters are:',mlp_best_params)
        self.params = mlp_best_params
        with open('best_params.py', 'w') as file:
            file.write(f"best_params = {mlp_best_params!r}\n")

        # Pretraining
        if self.pretrain_pct == 0.0:
            print('No pretraining')
        elif self.pretraining == True:
            print('with pretraining')
            mlp_predictor.pretrain(self.X_pretrain, self.Y_pretrain)
        else:
            print('same amount of training data no pretraining')

        # Training
        mlp_predictor.train(self.X_train, self.Y_train)

        # Prediction
        mlp_predictions = mlp_predictor.predict(self.X_test)

        # Metrics
        mlp_r2 = r2_score(self.Y_test, mlp_predictions)
        mlp_mse = mean_squared_error(self.Y_test, mlp_predictions)

        # Plot
        plt.scatter(self.Y_test, mlp_predictions)
        # Calculate the minimum and maximum values from both true and predicted values
        min_val = min(np.min(self.Y_test), np.min(mlp_predictions))
        max_val = max(np.max(self.Y_test), np.max(mlp_predictions))
        
        # Plot the Perfect Prediction Line (y = x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel("True Values (y_test)")
        plt.ylabel("Predicted Values (mlp_predictions)")
        plt.title("True vs Predicted Values")
        plt.savefig('NewNOn'f"True_vs_Predicted_{model}.png")
        plt.close()

        # Print Results
        print('For the mlp predictor the R^2 =', mlp_r2, '\nThe MSE =', mlp_mse)
        return mlp_r2, mlp_mse, mlp_best_params

if __name__ == "__main__":
    # Define file paths
    data_path = 'C:/Users/jesper/.spyder-py3/model_psnr_result.csv'
    flop_path = 'C:/Users/jesper/.spyder-py3/FLOPs.npy'
    param_path = 'C:/Users/jesper/.spyder-py3/params.npy'
    
    # parameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.15, 0.2]
    }

    # Parameter grid for SVR
    svr_param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'kernel': ['linear', 'rbf', 'poly']  # Kernel types
    }
    
    rbf_param_grid = {
        'kernel': ['cubic', 'tps'],  # Explore both cubic and thin-plate spline kernels
        'tail': ['linear', 'constant'],  # Try linear and constant tails
    }
    
    mlp_param_grid = {
    'n_layers': [1, 2, 3, 4],
    'n_hidden': [50, 100, 300, 500],
    'drop': [0.0, 0.1, 0.2, 0.4, 0.5],
    'lr': [0.005, 0.001, 0.0008, 0.0005]
    }
    
       # Initialize separate lists for each scenario
    results_with_pretraining = []
    results_no_pretraining = []
    results_full_no_pretraining = []
    
    # Initialize separate lists for best_params
    best_params_XGBoost_all = []
    best_params_SVR_all = []
    best_params_RBF_all = []
    best_params_MLP_all = []
    
    pretrain_pct = [0.0, 0.3]
    pretraining_options = [True, False]
    # Main loop
    for pct in pretrain_pct:
        if pct > 0.0:
            for prtrn in pretraining_options:
                # Initialize pipeline
                pipeline = PredictorPipeline(
                    data_path=data_path,
                    flop_path=flop_path,
                    param_path=param_path,
                    pretrain_pct=pct,
                    pretraining=prtrn,
                    remove_duplicates=False
                )
                
                if prtrn:
                    # Scenario 1: With Pretraining
                    logging.info("Running Scenario: With Pretraining")
                    
                    # XGBoost
                    start_time = time.time()
                    xgb_r2, xgb_mse, xgb_best_params = pipeline.run_XGBoostPipeline(param_grid=xgb_param_grid, model='XGB')
                    runtime_xgb = round(time.time() - start_time, 4)
                    results_with_pretraining.append({
                        "Model": "XGBoost",
                        "R^2": round(xgb_r2, 4),
                        "MSE": round(xgb_mse, 4),
                        "Runtime (s)": runtime_xgb
                    })
                    best_params_XGBoost_all.append(xgb_best_params)
                    
                    # SVR
                    start_time = time.time()
                    svr_r2, svr_mse, svr_best_params = pipeline.run_SVRPipeline(param_grid=svr_param_grid, model='SVR')
                    runtime_svr = round(time.time() - start_time, 4)
                    results_with_pretraining.append({
                        "Model": "SVR",
                        "R^2": round(svr_r2, 4),
                        "MSE": round(svr_mse, 4),
                        "Runtime (s)": runtime_svr
                    })
                    best_params_SVR_all.append(svr_best_params)
                    
                    # RBF
                    start_time = time.time()
                    rbf_r2, rbf_mse, rbf_best_params = pipeline.run_RBFPipeline(param_grid=rbf_param_grid, model='RBF')
                    runtime_rbf = round(time.time() - start_time, 4)
                    results_with_pretraining.append({
                        "Model": "RBF",
                        "R^2": round(rbf_r2, 4),
                        "MSE": round(rbf_mse, 4),
                        "Runtime (s)": runtime_rbf
                    })
                    best_params_RBF_all.append(rbf_best_params)
                    
                    # MLP
                    start_time = time.time()
                    mlp_r2, mlp_mse, mlp_best_params = pipeline.run_MLPPipeline(param_grid=mlp_param_grid, model='MLP')
                    runtime_mlp = round(time.time() - start_time, 4)
                    results_with_pretraining.append({
                        "Model": "MLP",
                        "R^2": round(mlp_r2, 4),
                        "MSE": round(mlp_mse, 4),
                        "Runtime (s)": runtime_mlp
                    })
                    best_params_MLP_all.append(mlp_best_params)
                    
                else:
                    # Scenario 2: With Same Amount of Training Data but No Pretraining
                    logging.info("Running Scenario: With Same Amount of Training Data but No Pretraining")
                    
                    # XGBoost
                    start_time = time.time()
                    xgb_r2, xgb_mse, xgb_best_params = pipeline.run_XGBoostPipeline(param_grid=xgb_param_grid, model='XGB')
                    runtime_xgb = round(time.time() - start_time, 4)
                    results_no_pretraining.append({
                        "Model": "XGBoost",
                        "R^2": round(xgb_r2, 4),
                        "MSE": round(xgb_mse, 4),
                        "Runtime (s)": runtime_xgb
                    })
                    best_params_XGBoost_all.append(xgb_best_params)
                    
                    # SVR
                    start_time = time.time()
                    svr_r2, svr_mse, svr_best_params = pipeline.run_SVRPipeline(param_grid=svr_param_grid, model='SVR')
                    runtime_svr = round(time.time() - start_time, 4)
                    results_no_pretraining.append({
                        "Model": "SVR",
                        "R^2": round(svr_r2, 4),
                        "MSE": round(svr_mse, 4),
                        "Runtime (s)": runtime_svr
                    })
                    best_params_SVR_all.append(svr_best_params)
                    
                    # RBF
                    start_time = time.time()
                    rbf_r2, rbf_mse, rbf_best_params = pipeline.run_RBFPipeline(param_grid=rbf_param_grid, model='RBF')
                    runtime_rbf = round(time.time() - start_time, 4)
                    results_no_pretraining.append({
                        "Model": "RBF",
                        "R^2": round(rbf_r2, 4),
                        "MSE": round(rbf_mse, 4),
                        "Runtime (s)": runtime_rbf
                    })
                    best_params_RBF_all.append(rbf_best_params)
                    
                    # MLP
                    start_time = time.time()
                    mlp_r2, mlp_mse, mlp_best_params = pipeline.run_MLPPipeline(param_grid=mlp_param_grid, model='MLP')
                    runtime_mlp = round(time.time() - start_time, 4)
                    results_no_pretraining.append({
                        "Model": "MLP",
                        "R^2": round(mlp_r2, 4),
                        "MSE": round(mlp_mse, 4),
                        "Runtime (s)": runtime_mlp
                    })
                    best_params_MLP_all.append(mlp_best_params)
        else:
            # Scenario 3: With Full Data Set and No Pretraining
            logging.info("Running Scenario: With Full Data Set and No Pretraining")
            # Initialize pipeline without removing duplicates
            pipeline = PredictorPipeline(
                data_path=data_path,
                flop_path=flop_path,
                param_path=param_path,
                pretrain_pct=pct,
                pretraining=False,  # Set to False since pretrain_pct=0.0
                remove_duplicates=False  # Do not remove duplicates
            )
            
            # Run XGBoost
            start_time = time.time()
            xgb_r2, xgb_mse, xgb_best_params = pipeline.run_XGBoostPipeline(param_grid=xgb_param_grid, model='XGB')
            runtime_xgb = round(time.time() - start_time, 4)
            results_full_no_pretraining.append({
                "Model": "XGBoost",
                "R^2": round(xgb_r2, 4),
                "MSE": round(xgb_mse, 4),
                "Runtime (s)": runtime_xgb
            })
            best_params_XGBoost_all.append(xgb_best_params)
            
            # Run SVR
            start_time = time.time()
            svr_r2, svr_mse, svr_best_params = pipeline.run_SVRPipeline(param_grid=svr_param_grid, model='SVR')
            runtime_svr = round(time.time() - start_time, 4)
            results_full_no_pretraining.append({
                "Model": "SVR",
                "R^2": round(svr_r2, 4),
                "MSE": round(svr_mse, 4),
                "Runtime (s)": runtime_svr
            })
            best_params_SVR_all.append(svr_best_params)
            
            # Run RBF
            start_time = time.time()
            rbf_r2, rbf_mse, rbf_best_params = pipeline.run_RBFPipeline(param_grid=rbf_param_grid, model='RBF')
            runtime_rbf = round(time.time() - start_time, 4)
            results_full_no_pretraining.append({
                "Model": "RBF",
                "R^2": round(rbf_r2, 4),
                "MSE": round(rbf_mse, 4),
                "Runtime (s)": runtime_rbf
            })
            best_params_RBF_all.append(rbf_best_params)
            
            # Run MLP
            start_time = time.time()
            mlp_r2, mlp_mse, mlp_best_params = pipeline.run_MLPPipeline(param_grid=mlp_param_grid, model='MLP')
            runtime_mlp = round(time.time() - start_time, 4)
            results_full_no_pretraining.append({
                "Model": "MLP",
                "R^2": round(mlp_r2, 4),
                "MSE": round(mlp_mse, 4),
                "Runtime (s)": runtime_mlp
            })
            best_params_MLP_all.append(mlp_best_params)
        
    # Create DataFrames from each scenario's results
    df_with_pretraining = pd.DataFrame(results_with_pretraining)
    df_no_pretraining = pd.DataFrame(results_no_pretraining)
    df_full_no_pretraining = pd.DataFrame(results_full_no_pretraining)
    
    # Save each DataFrame to a separate CSV file
    df_with_pretraining.to_csv('NewNOnpropersplit_with_pretraining.csv', index=False)
    df_no_pretraining.to_csv('NewNOnpropersplit_no_pretraining.csv', index=False)
    df_full_no_pretraining.to_csv('NewNOnpropersplit_full_no_pretraining.csv', index=False)
    
    # Save best_params to separate CSV files
    df_best_params_XGBoost = pd.DataFrame(best_params_XGBoost_all)
    df_best_params_SVR = pd.DataFrame(best_params_SVR_all)
    df_best_params_RBF = pd.DataFrame(best_params_RBF_all)
    df_best_params_MLP = pd.DataFrame(best_params_MLP_all)
    
    df_best_params_XGBoost.to_csv('Nonbest_params_XGBoost.csv', index=False)
    df_best_params_SVR.to_csv('Nonbest_params_SVR.csv', index=False)
    df_best_params_RBF.to_csv('Nonbest_params_RBF.csv', index=False)
    df_best_params_MLP.to_csv('Nonbest_params_MLP.csv', index=False)
    
    # Optional: Print the DataFrames for verification
    print("\nResults with Pretraining:")
    print(df_with_pretraining)
    
    print("\nResults with Same Amount of Training Data but No Pretraining:")
    print(df_no_pretraining)
    
    print("\nResults with Full Data Set and No Pretraining:")
    print(df_full_no_pretraining)
    
    print("\nBest Parameters for XGBoost:")
    print(df_best_params_XGBoost)
    
    print("\nBest Parameters for SVR:")
    print(df_best_params_SVR)
    
    print("\nBest Parameters for RBF:")
    print(df_best_params_RBF)
    
    print("\nBest Parameters for MLP:")
    print(df_best_params_MLP)
