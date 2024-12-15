import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from WindPy import w
import datetime
import time
import statsmodels.api as sm
import os
import re

# -----------------------------------------
# Functions
# -----------------------------------------

def load_esg_data(esg_predictions_path):
    """
    Load ESG predictions from CSV.
    """
    esg_df = pd.read_csv(esg_predictions_path)

    return esg_df



def aggregate_esg_metrics(esg_df):
    """
    Aggregate ESG metrics by calculating ESG Score and total words per company.
    """
    # ESG Score: Total number of sentences per company
    esg_score = esg_df.groupby('Source PDF').size().reset_index(name='ESG Score')
    
    # Total Words: Total number of words per company
    esg_df['Word Count'] = esg_df['Original Text'].apply(lambda x: len(re.findall(r'\w+', str(x))))
    total_words = esg_df.groupby('Source PDF')['Word Count'].sum().reset_index(name='Total Words')
    
    # Merge ESG Score and Total Words
    aggregated = pd.merge(esg_score, total_words, on='Source PDF')
    
    # Calculate Normalized ESG Score
    aggregated['Normalized ESG Score'] = aggregated['ESG Score'] / aggregated['Total Words']
    
    # Rename 'Source PDF' to 'Company' for clarity
    aggregated.rename(columns={'Source PDF': 'Company'}, inplace=True)
    

    return aggregated



def get_stock_price_change(company_code, start_date, end_date):
    """
    Retrieves the percentage change in stock price for a given company between start_date and end_date.
    
    Parameters:
        company_code (str): Wind financial code of the company.
        start_date (str): Start date in 'YYYYMMDD' format.
        end_date (str): End date in 'YYYYMMDD' format.
    
    Returns:
        float: Percentage change in stock price.
    """
    # Retrieve closing prices

    data = w.wsd(company_code, "close", start_date, end_date, "PriceAdj=F")
    
    if data.ErrorCode != 0:
        print(f"Error retrieving data for {company_code}: {data.ErrorMsg}")
        return None
    
    # Extract closing prices
    closes = data.Data[0]
    
    if len(closes) < 2:
        print(f"Insufficient data for {company_code}.")
        return None
    
    # Calculate percentage change
    price_change = ((closes[-1] - closes[0]) / closes[0]) * 100

    return price_change


def retrieve_stock_data(companies, start_date, end_date):
    """
    Retrieves stock price changes for a list of companies.
    """
    stock_changes = []
    for company in companies:
        print(f"Retrieving stock data for: {company}")
        change = get_stock_price_change(company, start_date, end_date)
        stock_changes.append(change)
        time.sleep(0.1)  # To prevent overwhelming the API

    return stock_changes



def prepare_regression_data(aggregated_esg, stock_changes):
    """
    Merge ESG metrics with stock price changes.
    """
    aggregated_esg['Stock Price Change (%)'] = stock_changes
    # Drop companies with missing stock data
    aggregated_esg.dropna(subset=['Stock Price Change (%)'], inplace=True)
    return aggregated_esg



def perform_regression_analysis(X, y):
    """
    Perform linear regression analysis.
    """
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model



def create_scatter_plot(independent_var, dependent_var, plot_df, output_dir):
    """
    Create and save scatter plot with regression line.
    """
    plt.figure(figsize=(8,6))
    sns.regplot(x=independent_var, y=dependent_var, data=plot_df, scatter_kws={'alpha':0.5})
    plt.title(f"{independent_var} vs {dependent_var}")
    plt.xlabel(independent_var)
    plt.ylabel(dependent_var)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{independent_var}_vs_{dependent_var}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Scatter plot saved to {plot_path}")


def create_residual_plot(model, X, y, output_dir):
    """
    Create and save residuals vs fitted values plot.
    """
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    
    # Save the residual plot
    residual_plot_filename = "residuals_vs_fitted.png"
    residual_plot_path = os.path.join(output_dir, residual_plot_filename)
    plt.savefig(residual_plot_path)
    plt.close()
    print(f"Residuals plot saved to {residual_plot_path}")

def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths
    esg_predictions_path = "supervised_learning_output/unlabeled_predictions.csv"
    regression_output_dir = "regression_output"
    
    # Dates
    start_date = datetime.datetime(2022, 12, 1)
    end_date = datetime.datetime(2024, 12, 1)
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    # -------------------------------
    # Initialize WindPy
    # -------------------------------
    
    print("Initializing WindPy...")
    w.start()
    if not w.isconnected():
        print("WindPy not connected. Please check your Wind installation and connection.")
        return
    
    # -------------------------------
    # Load and Aggregate ESG Metrics
    # -------------------------------
    
    print("Loading ESG predictions...")
    esg_df = load_esg_data(esg_predictions_path)
    print(f"Total ESG records loaded: {len(esg_df)}")
    
    print("Aggregating ESG metrics...")
    aggregated_esg = aggregate_esg_metrics(esg_df)
    print(f"Aggregated ESG metrics:\n{aggregated_esg.head()}")
    
    # -------------------------------
    # Retrieve Stock Price Data
    # -------------------------------
    
    companies = aggregated_esg['Company'].tolist()
    print("Retrieving stock price changes...")
    stock_changes = retrieve_stock_data(companies, start_date_str, end_date_str)
    
    # -------------------------------
    # Prepare Regression Data
    # -------------------------------
    
    print("Preparing regression data...")
    regression_df = prepare_regression_data(aggregated_esg, stock_changes)
    print(f"Regression data prepared:\n{regression_df.head()}")
    
    # -------------------------------
    # Perform Regression Analysis
    # -------------------------------
    
    print("Performing regression analysis...")
    X = regression_df[['ESG Score', 'Normalized ESG Score']]
    y = regression_df['Stock Price Change (%)']
    model = perform_regression_analysis(X, y)
    
    # Save regression summary to a text file
    summary_filename = "regression_summary.txt"
    summary_path = os.path.join(regression_output_dir, summary_filename)
    os.makedirs(regression_output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(model.summary().as_text())
    print(f"Regression summary saved to {summary_path}")
    
    # -------------------------------
    # Create Visualizations
    # -------------------------------
    
    # Create scatter plots for ESG Score and Normalized ESG Score
    for independent_var in ['ESG Score', 'Normalized ESG Score']:
        print(f"Creating scatter plot for {independent_var}...")
        plot_df = regression_df[[independent_var, 'Stock Price Change (%)']]
        create_scatter_plot(independent_var, 'Stock Price Change (%)', plot_df, regression_output_dir)
    
    # Create residual plot
    print("Creating residual plot...")
    create_residual_plot(model, X, y, regression_output_dir)
    
    print("Regression analysis complete!")

if __name__ == "__main__":
    main()
