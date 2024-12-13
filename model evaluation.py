import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_classification_model(y_true, y_pred, output_filepath):
    """
    Evaluates a classification model with metrics and saves a confusion matrix plot.
    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        output_filepath (str): Path to save the confusion matrix plot.
    """
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_filepath)
    plt.show()

def evaluate_regression_model(y_true, y_pred, output_filepath):
    """
    Evaluates a regression model with metrics and visualizations.
    Args:
        y_true (array): True values.
        y_pred (array): Predicted values.
        output_filepath (str): Path to save the residual plot.
    """
    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(output_filepath)
    plt.show()

if __name__ == "__main__":
    # Filepaths
    classification_data_filepath = "classification_results.csv"  # Replace with your file
    regression_data_filepath = "regression_results.csv"          # Replace with your file

    # Classification Model Evaluation
    print("Evaluating Classification Model...")
    classification_data = pd.read_csv(classification_data_filepath)
    y_true_classification = classification_data["true_labels"]
    y_pred_classification = classification_data["predicted_labels"]
    evaluate_classification_model(y_true_classification, y_pred_classification, "confusion_matrix.png")

    # Regression Model Evaluation
    print("Evaluating Regression Model...")
    regression_data = pd.read_csv(regression_data_filepath)
    y_true_regression = regression_data["true_values"]
    y_pred_regression = regression_data["predicted_values"]
    evaluate_regression_model(y_true_regression, y_pred_regression, "residual_plot.png")
