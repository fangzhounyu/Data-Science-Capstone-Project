import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import label_binarize



def load_model(model_path):
    """
    Load a trained model from a file.

    """
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None



def load_test_data(test_data_path):
    """
    Load the preprocessed test data.

    """
    try:
        test_df = pd.read_csv(test_data_path)
        print(f"Successfully loaded test data from {test_data_path}")
        return test_df
    except FileNotFoundError:
        print(f"Test data file not found: {test_data_path}")
        return None

def evaluate_classification_model(model, X_test, y_test, output_dir):
    """
    Evaluate the classification model and generate evaluation metrics and plots.

    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve and AUC
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, thresholds, roc_auc = None, None, None, None
    
    # Save metrics to a dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Classification Report": report,
        "Confusion Matrix": cm,
        "FPR": fpr,
        "TPR": tpr,
        "ROC AUC": roc_auc,
    }
    
    # Create Evaluation Directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Classification Report
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
    

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    
    # Plot ROC Curve
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"Saved ROC curve plot to {roc_plot_path}")
    
    return metrics



def evaluate_regression_model(model, X_test, y_test, output_dir):
    """
    Evaluate the regression model and generate evaluation metrics and plots.
    
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save metrics to a dictionary
    metrics = {
        "R-squared": r2,
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
    }
    
    # Create Evaluation Directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print and Save Metrics
    metrics_path = os.path.join(output_dir, "regression_metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved regression metrics to {metrics_path}")
    
    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    residual_plot_path = os.path.join(output_dir, "residuals_vs_fitted.png")
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Saved residuals plot to {residual_plot_path}")
    
    # Prediction vs Actual Scatter Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title("Prediction vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    prediction_plot_path = os.path.join(output_dir, "prediction_vs_actual.png")
    plt.savefig(prediction_plot_path)
    plt.close()
    print(f"Saved prediction vs actual plot to {prediction_plot_path}")
    
    return metrics



def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths to Models
    supervised_model_path = "models/supervised_learning_model.joblib"  # Adjust path as needed
    regression_model_path = "models/regression_model.joblib"          # Adjust path as needed
    
    # Path to Test Data
    test_data_path = "data/test_data.csv"  # Adjust path as needed
    
    # Output Directory for Evaluation
    evaluation_output_dir = "model_evaluation_output"
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    # -------------------------------
    # Load Trained Models
    # -------------------------------
    
    print("Loading trained models...")
    supervised_model = load_model(supervised_model_path)
    regression_model = load_model(regression_model_path)
    
    if supervised_model is None or regression_model is None:
        print("One or both models could not be loaded. Exiting evaluation.")
        return
    
    # -------------------------------
    # Load Test Data
    # -------------------------------
    
    print("Loading test data...")
    test_df = load_test_data(test_data_path)
    if test_df is None:
        print("Test data could not be loaded. Exiting evaluation.")
        return
    
    # -------------------------------
    # Prepare Data for Classification Evaluation
    # -------------------------------
    
    # Assuming the supervised learning model was trained to predict 'ESG Category'
    # and the test data contains 'ESG Category' as the target.
    
    if 'ESG Category' not in test_df.columns:
        print("ESG Category column not found in test data for classification evaluation.")
        classification_evaluation = None
    else:
        X_test_class = test_df.drop(columns=['ESG Category'])
        y_test_class = test_df['ESG Category']
        
        # Evaluate Classification Model
        print("\nEvaluating Classification Model...")
        classification_metrics = evaluate_classification_model(
            model=supervised_model,
            X_test=X_test_class,
            y_test=y_test_class,
            output_dir=os.path.join(evaluation_output_dir, "classification")
        )
        print("\nClassification Evaluation Metrics:")
        for key, value in classification_metrics.items():
            if key not in ["Confusion Matrix", "FPR", "TPR", "ROC AUC"]:
                print(f"{key}: {value}")
    
    # -------------------------------
    # Prepare Data for Regression Evaluation
    # -------------------------------
    
    # Assuming the regression model was trained to predict 'Stock Price Change (%)'
    if 'Stock Price Change (%)' not in test_df.columns:
        print("Stock Price Change (%) column not found in test data for regression evaluation.")
        regression_evaluation = None
    else:
        X_test_reg = test_df.drop(columns=['Stock Price Change (%)'])
        y_test_reg = test_df['Stock Price Change (%)']
        
        # Evaluate Regression Model
        print("\nEvaluating Regression Model...")
        regression_metrics = evaluate_regression_model(
            model=regression_model,
            X_test=X_test_reg,
            y_test=y_test_reg,
            output_dir=os.path.join(evaluation_output_dir, "regression")
        )
        print("\nRegression Evaluation Metrics:")
        for key, value in regression_metrics.items():
            print(f"{key}: {value}")
    
    print("\nModel Evaluation complete! All evaluation results and plots are saved in the 'model_evaluation_output/' directory.")

if __name__ == "__main__":
    main()
