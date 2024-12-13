import pandas as pd
from climatebert_processing import process_dataset as process_climatebert
from supervised_learning import train_model, evaluate_model, preprocess_for_training
from regression_analysis import perform_regression, plot_regression_results, evaluate_model as evaluate_regression
from eda_visualization import generate_summary_statistics, plot_distribution, plot_correlation_heatmap

# Paths
RAW_DATA_PATH = "raw_data.csv"
CLIMATEBERT_OUTPUT_PATH = "climatebert_output.csv"
LABELED_DATA_PATH = "labeled_data.csv"
REGRESSION_DATA_PATH = "regression_data.csv"
MODEL_OUTPUT_PATH = "random_forest_model.pkl"
VECTORIZER_OUTPUT_PATH = "tfidf_vectorizer.pkl"
REGRESSION_PLOT_PATH = "regression_plot.png"

def run_eda(data):
    """
    Perform exploratory data analysis on the dataset.
    Args:
        data (DataFrame): Dataset to analyze.
    """
    print("Running EDA...")
    generate_summary_statistics(data)
    plot_distribution(data, "esg_score", "eda_esg_score_distribution.png")
    plot_correlation_heatmap(data, "eda_correlation_heatmap.png")

def run_climatebert_processing():
    """
    Run ClimateBERT processing on the raw dataset.
    """
    print("Running ClimateBERT processing...")
    process_climatebert(RAW_DATA_PATH, CLIMATEBERT_OUTPUT_PATH)

def run_supervised_learning():
    """
    Train and evaluate the supervised learning model.
    """
    print("Running supervised learning...")
    data = pd.read_csv(LABELED_DATA_PATH)
    text_column = "cleaned_text"
    label_column = "esg_category"

    # Preprocess data for training
    X_train, X_test, y_train, y_test, vectorizer = preprocess_for_training(data, text_column, label_column)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model and vectorizer
    vectorizer.dump(VECTORIZER_OUTPUT_PATH)
    model.dump(MODEL_OUTPUT_PATH)
    print("Supervised learning model and vectorizer saved.")

def run_regression_analysis():
    """
    Perform regression analysis and plot results.
    """
    print("Running regression analysis...")
    data = pd.read_csv(REGRESSION_DATA_PATH)
    esg_columns = ["metric_1", "metric_2", "metric_3"]
    target_column = "stock_price_change"

    # Prepare data
    X, y = preprocess_data(data, esg_columns, target_column)

    # Perform regression
    model = perform_regression(X, y)
    print(model.summary())

    # Predict and evaluate
    predictions = model.predict(X)
    evaluate_regression(y, predictions)

    # Plot results
    plot_regression_results(X, y, predictions, REGRESSION_PLOT_PATH)

if __name__ == "__main__":
    print("Integration pipeline started...")

    # Step 1: EDA
    raw_data = pd.read_csv(RAW_DATA_PATH)
    run_eda(raw_data)

    # Step 2: ClimateBERT Processing
    run_climatebert_processing()

    # Step 3: Supervised Learning
    run_supervised_learning()

    # Step 4: Regression Analysis
    run_regression_analysis()

    print("Pipeline completed successfully!")
