import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def load_training_data(train_data_path):
    """
    Load the preprocessed training data.
  
    """
    try:
        train_df = pd.read_csv(train_data_path)
        print(f"Successfully loaded training data from {train_data_path}")
        return train_df
    except FileNotFoundError:
        print(f"Training data file not found: {train_data_path}")
        return None



def prepare_classification_data(train_df):
    """
    Prepare features and target for classification.
    
    """
    if 'ESG Category' not in train_df.columns:
        print("ESG Category column not found in training data.")
        return None, None
    X = train_df.drop(columns=['ESG Category'])
    y = train_df['ESG Category']
    return X, y



def prepare_regression_data(train_df):
    """
    Prepare features and target for regression.
    
    """
    if 'Stock Price Change (%)' not in train_df.columns:
        print("Stock Price Change (%) column not found in training data.")
        return None, None
    X = train_df.drop(columns=['Stock Price Change (%)'])
    y = train_df['Stock Price Change (%)']
    return X, y


def define_hyperparameter_grids():
    """
    Define hyperparameter grids for classification and regression models.
    
    Returns:
        dict: Dictionary containing hyperparameter grids.
    """
    hyperparameter_grids = {
        'classification': {
            'RandomForestClassifier': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__bootstrap': [True, False]
            },
            'LogisticRegression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            },
            'SVC': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__kernel': ['rbf', 'linear']
            }
        },
        'regression': {
            'RandomForestRegressor': {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__bootstrap': [True, False]
            },
            'Ridge': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
                'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
            },
            'Lasso': {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
                'regressor__selection': ['cyclic', 'random']
            },
            'SVR': {
                'regressor__C': [0.1, 1, 10, 100],
                'regressor__gamma': ['scale', 'auto'],
                'regressor__kernel': ['rbf', 'linear']
            }
        }
    }
    return hyperparameter_grids


def perform_grid_search(model, param_grid, X, y, cv=5, scoring=None):
    """
    Perform GridSearchCV to find the best hyperparameters.
    
    
    Returns:
        GridSearchCV: Fitted GridSearchCV object.
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)
    return grid_search

def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths
    train_data_path = "data/train_data.csv"  # Adjust path as needed
    models_dir = "models"
    tuned_models_dir = os.path.join(models_dir, "tuned_models")
    os.makedirs(tuned_models_dir, exist_ok=True)
    tuning_results_dir = "hyperparameter_tuning_output"
    os.makedirs(tuning_results_dir, exist_ok=True)
    
    # -------------------------------
    # Load Training Data
    # -------------------------------
    
    print("Loading training data...")
    train_df = load_training_data(train_data_path)
    if train_df is None:
        return
    
    # -------------------------------
    # Prepare Data for Classification
    # -------------------------------
    
    print("\nPreparing data for classification...")
    X_class, y_class = prepare_classification_data(train_df)
    if X_class is None or y_class is None:
        print("Failed to prepare classification data. Skipping classification tuning.")
        classification_tuned_model = None
    else:
        # -------------------------------
        # Define Pipeline for Classification
        # -------------------------------
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Example with RandomForestClassifier
        classification_pipeline_rf = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Example with LogisticRegression
        classification_pipeline_lr = Pipeline([
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Example with SVC
        classification_pipeline_svc = Pipeline([
            ('classifier', SVC(probability=True, random_state=42))
        ])
        
        # -------------------------------
        # Define Hyperparameter Grids
        # -------------------------------
        
        hyperparameter_grids = define_hyperparameter_grids()
        classification_grids = hyperparameter_grids['classification']
        
        # -------------------------------
        # Perform Grid Search for Classification Models
        # -------------------------------
        
        tuned_classification_models = {}
        for clf_name, param_grid in classification_grids.items():
            print(f"\nPerforming Grid Search for {clf_name}...")
            if clf_name == 'RandomForestClassifier':
                pipeline = classification_pipeline_rf
            elif clf_name == 'LogisticRegression':
                pipeline = classification_pipeline_lr
            elif clf_name == 'SVC':
                pipeline = classification_pipeline_svc
            else:
                print(f"Unknown classifier: {clf_name}")
                continue
            
            grid_search = perform_grid_search(
                model=pipeline,
                param_grid=param_grid,
                X=X_class,
                y=y_class,
                cv=5,
                scoring='accuracy'
            )
            
            print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            
            # Save the best model
            tuned_model_path = os.path.join(tuned_models_dir, f"{clf_name}_tuned.joblib")
            joblib.dump(grid_search.best_estimator_, tuned_model_path)
            print(f"Saved tuned {clf_name} model to {tuned_model_path}")
            
            # Store the best model
            tuned_classification_models[clf_name] = grid_search.best_estimator_
        
        print("\nClassification Hyperparameter Tuning Complete!")
    
    # -------------------------------
    # Prepare Data for Regression
    # -------------------------------
    
    print("\nPreparing data for regression...")
    X_reg, y_reg = prepare_regression_data(train_df)
    if X_reg is None or y_reg is None:
        print("Failed to prepare regression data. Skipping regression tuning.")
        regression_tuned_model = None
    else:
        # -------------------------------
        # Define Pipeline for Regression
        # -------------------------------
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Example with RandomForestRegressor
        regression_pipeline_rf = Pipeline([
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Example with Ridge
        regression_pipeline_ridge = Pipeline([
            ('regressor', Ridge(random_state=42))
        ])
        
        # Example with Lasso
        regression_pipeline_lasso = Pipeline([
            ('regressor', Lasso(random_state=42))
        ])
        
        # Example with SVR
        regression_pipeline_svr = Pipeline([
            ('regressor', SVR())
        ])
        
        # -------------------------------
        # Define Hyperparameter Grids
        # -------------------------------
        
        regression_grids = hyperparameter_grids['regression']
        
        # -------------------------------
        # Perform Grid Search for Regression Models
        # -------------------------------
        
        tuned_regression_models = {}
        for reg_name, param_grid in regression_grids.items():
            print(f"\nPerforming Grid Search for {reg_name}...")
            if reg_name == 'RandomForestRegressor':
                pipeline = regression_pipeline_rf
            elif reg_name == 'Ridge':
                pipeline = regression_pipeline_ridge
            elif reg_name == 'Lasso':
                pipeline = regression_pipeline_lasso
            elif reg_name == 'SVR':
                pipeline = regression_pipeline_svr
            else:
                print(f"Unknown regressor: {reg_name}")
                continue
            
            grid_search = perform_grid_search(
                model=pipeline,
                param_grid=param_grid,
                X=X_reg,
                y=y_reg,
                cv=5,
                scoring='r2'
            )
            
            print(f"Best parameters for {reg_name}: {grid_search.best_params_}")
            print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
            
            # Save the best model
            tuned_model_path = os.path.join(tuned_models_dir, f"{reg_name}_tuned.joblib")
            joblib.dump(grid_search.best_estimator_, tuned_model_path)
            print(f"Saved tuned {reg_name} model to {tuned_model_path}")
            
            # Store the best model
            tuned_regression_models[reg_name] = grid_search.best_estimator_
        
        print("\nRegression Hyperparameter Tuning Complete!")
    
    # -------------------------------
    # Save Tuning Results
    # -------------------------------
    
    print("\nSaving hyperparameter tuning results...")
    results_path = os.path.join(tuning_results_dir, "hyperparameter_tuning_results.csv")
    
    # Collect Classification Results
    classification_results = []
    for clf_name, model in tuned_classification_models.items():
        best_params = model.named_steps['classifier'].get_params()
        classification_results.append({
            'Model': clf_name,
            'Best Parameters': best_params,
            'Best Score': model.score(X_class, y_class)
        })
    
    # Collect Regression Results
    regression_results = []
    for reg_name, model in tuned_regression_models.items():
        best_params = model.named_steps['regressor'].get_params()
        regression_results.append({
            'Model': reg_name,
            'Best Parameters': best_params,
            'Best Score': model.score(X_reg, y_reg)
        })
    
    # Create DataFrames
    classification_df = pd.DataFrame(classification_results)
    regression_df = pd.DataFrame(regression_results)
    
    # Write to CSV
    with open(results_path, "w") as f:
        f.write("Classification Models\n")
        classification_df.to_csv(f, index=False)
        f.write("\nRegression Models\n")
        regression_df.to_csv(f, index=False)
    
    print(f"Saved hyperparameter tuning results to {results_path}")
    
    print("\nHyperparameter Tuning Complete! Tuned models are saved in the 'models/tuned_models/' directory.")
    
if __name__ == "__main__":
    main()
