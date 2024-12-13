import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """
    Loads the dataset for regression analysis.
    Args:
        filepath (str): Path to the dataset.
    Returns:
        pandas.DataFrame: Dataset containing ESG metrics and stock prices.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, esg_columns, target_column):
    """
    Prepares data for regression by summing ESG metrics and adding intercept for regression.
    Args:
        data (pandas.DataFrame): Input dataset.
        esg_columns (list): List of columns containing ESG metrics.
        target_column (str): Name of the target variable (e.g., stock price change).
    Returns:
        tuple: Features (X) and target (y) for regression.
    """
    # Aggregate ESG metrics
    data['esg_score'] = data[esg_columns].sum(axis=1)
    X = data[['esg_score']]  # Feature: aggregated ESG score
    X = sm.add_constant(X)   # Add intercept for regression
    y = data[target_column]  # Target: stock price change
    return X, y

def perform_regression(X, y):
    """
    Performs linear regression analysis.
    Args:
        X (DataFrame): Features for regression.
        y (Series): Target variable.
    Returns:
        RegressionResults: Fitted regression model.
    """
    model = sm.OLS(y, X).fit()
    return model

def plot_regression_results(X, y, predictions, output_filepath):
    """
    Plots the scatter plot with regression line.
    Args:
        X (DataFrame): Features used for regression.
        y (Series): Actual target values.
        predictions (Series): Predicted values from the regression.
        output_filepath (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X['esg_score'], y, color='blue', alpha=0.6, label="Actual")
    plt.plot(X['esg_score'], predictions, color='red', linewidth=2, label="Regression Line")
    plt.xlabel("Aggregated ESG Score")
    plt.ylabel("Stock Price Change (%)")
    plt.title("Regression Analysis: ESG Score vs. Stock Price Change")
    plt.legend()
    plt.savefig(output_filepath)
    plt.show()

def evaluate_model(y, predictions):
    """
    Evaluates the regression model with metrics like R² and RMSE.
    Args:
        y (Series): Actual target values.
        predictions (Series): Predicted values from the regression.
    """
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    # Filepath to the dataset
    dataset_filepath = "regression_data.csv"  # Replace with your actual dataset
    output_plot = "regression_plot.png"

    # Columns for ESG metrics and stock price change
    esg_columns = ["metric_1", "metric_2", "metric_3"]  # Replace with actual ESG metric columns
    target_column = "stock_price_change"

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_data(dataset_filepath)
    X, y = preprocess_data(data, esg_columns, target_column)

    # Perform regression
    print("Performing regression analysis...")
    model = perform_regression(X, y)

    # Print regression summary
    print(model.summary())

    # Predict and evaluate
    predictions = model.predict(X)
    evaluate_model(y, predictions)

    # Plot results
    print("Plotting regression results...")
    plot_regression_results(X, y, predictions, output_plot)
    print(f"Regression plot saved to {output_plot}!")
