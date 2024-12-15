import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

#implementary

def load_esg_data(esg_predictions_path):
    """
    Load ESG predictions from CSV.
    
    Parameters:
        esg_predictions_path (str): Path to the ESG predictions CSV file.
    
    Returns:
        pd.DataFrame: Loaded ESG predictions data.
    """
    try:
        esg_df = pd.read_csv(esg_predictions_path)
        print(f"Successfully loaded data from {esg_predictions_path}")
        return esg_df
    
    except FileNotFoundError:
        print(f"File not found: {esg_predictions_path}")
        return None



def display_basic_statistics(esg_df):
    """
    Display basic statistics and information about the ESG DataFrame.
    
    Parameters:
        esg_df (pd.DataFrame): ESG predictions data.
    """
    print("\n=== Basic Statistics ===")
    print(esg_df.describe(include='all'))
    
    print("\n=== Data Types ===")
    print(esg_df.dtypes)
    
    print("\n=== Missing Values ===")
    print(esg_df.isnull().sum())



def aggregate_esg_metrics(esg_df):
    """
    Aggregate ESG metrics by calculating ESG Score and Total Words per company.
    
    Parameters:
        esg_df (pd.DataFrame): ESG predictions data.
    
    Returns:
        pd.DataFrame: Aggregated ESG metrics.
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
    
    print("\n=== Aggregated ESG Metrics ===")
    print(aggregated.head())
    

    return aggregated



def plot_distribution(data, column, title, xlabel, ylabel, output_path):
    
    plt.figure(figsize=(8,6))
    sns.histplot(data[column], kde=True, bins=30, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved distribution plot: {output_path}")



def plot_bar_chart(data, column, title, xlabel, ylabel, output_path):
    """
    Plot and save a bar chart for a categorical column.
    
    """
    plt.figure(figsize=(10,6))
    sns.countplot(data=data, x=column, palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved bar chart: {output_path}")



def plot_correlation_heatmap(data, title, output_path):
    """
    Plot and save a correlation heatmap.

    """
    plt.figure(figsize=(10,8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved correlation heatmap: {output_path}")



def plot_scatter(data, x, y, title, xlabel, ylabel, output_path):
    """
    Plot and save a scatter plot with regression line.

    """
    plt.figure(figsize=(8,6))
    sns.regplot(data=data, x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved scatter plot: {output_path}")



def plot_boxplot(data, column, title, ylabel, output_path):
    """
    Plot and save a box plot for a numerical column.
   
    """
    plt.figure(figsize=(8,6))
    sns.boxplot(data=data, y=column, color='lightgreen')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved box plot: {output_path}")



def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths
    esg_predictions_path = "supervised_learning_output/unlabeled_predictions.csv"
    eda_output_dir = "eda_visualization_output"
    os.makedirs(eda_output_dir, exist_ok=True)
    
    # -------------------------------
    # Load ESG Data
    # -------------------------------
    
    print("Loading ESG predictions...")
    esg_df = load_esg_data(esg_predictions_path)
    if esg_df is None:
        return
    
    # -------------------------------
    # Display Basic Statistics
    # -------------------------------
    
    display_basic_statistics(esg_df)
    
    # -------------------------------
    # Aggregate ESG Metrics
    # -------------------------------
    
    aggregated_esg = aggregate_esg_metrics(esg_df)
    
    # -------------------------------
    # Visualize ESG Category Distribution
    # -------------------------------
    
    print("\nPlotting ESG Category Distribution...")
    esg_category_plot_path = os.path.join(eda_output_dir, "esg_category_distribution.png")
    plot_bar_chart(
        data=esg_df,
        column='Predicted ESG Category',
        title='ESG Category Distribution',
        xlabel='ESG Category',
        ylabel='Number of Sentences',
        output_path=esg_category_plot_path
    )
    
    # -------------------------------
    # Visualize ESG Score Distribution
    # -------------------------------
    
    print("\nPlotting ESG Score Distribution...")
    esg_score_plot_path = os.path.join(eda_output_dir, "esg_score_distribution.png")
    plot_distribution(
        data=aggregated_esg,
        column='ESG Score',
        title='Distribution of ESG Scores',
        xlabel='ESG Score (Total Sentences)',
        ylabel='Frequency',
        output_path=esg_score_plot_path
    )
    
    # -------------------------------
    # Visualize Normalized ESG Score Distribution
    # -------------------------------
    
    print("\nPlotting Normalized ESG Score Distribution...")
    normalized_esg_score_plot_path = os.path.join(eda_output_dir, "normalized_esg_score_distribution.png")
    plot_distribution(
        data=aggregated_esg,
        column='Normalized ESG Score',
        title='Distribution of Normalized ESG Scores',
        xlabel='Normalized ESG Score (ESG Score / Total Words)',
        ylabel='Frequency',
        output_path=normalized_esg_score_plot_path
    )
    
    # -------------------------------
    # Visualize Prediction Confidence Distribution
    # -------------------------------
    
    print("\nPlotting Prediction Confidence Distribution...")
    prediction_confidence_plot_path = os.path.join(eda_output_dir, "prediction_confidence_distribution.png")
    plot_distribution(
        data=esg_df,
        column='Prediction Confidence',
        title='Distribution of Prediction Confidence',
        xlabel='Prediction Confidence',
        ylabel='Frequency',
        output_path=prediction_confidence_plot_path
    )
    
    # -------------------------------
    # Visualize Word Count Distribution
    # -------------------------------
    
    print("\nPlotting Word Count Distribution...")
    word_count_plot_path = os.path.join(eda_output_dir, "word_count_distribution.png")
    plot_distribution(
        data=esg_df,
        column='Word Count',
        title='Distribution of Word Counts',
        xlabel='Word Count',
        ylabel='Frequency',
        output_path=word_count_plot_path
    )
    
    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    
    print("\nPlotting Correlation Heatmap...")
    numerical_data = aggregated_esg[['ESG Score', 'Total Words', 'Normalized ESG Score']]
    correlation_heatmap_path = os.path.join(eda_output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(
        data=numerical_data,
        title='Correlation Heatmap of ESG Metrics',
        output_path=correlation_heatmap_path
    )
    
    # -------------------------------
    # Scatter Plot: ESG Score vs Prediction Confidence
    # -------------------------------
    
    print("\nPlotting Scatter Plot: ESG Score vs Prediction Confidence...")
    scatter_plot_path = os.path.join(eda_output_dir, "esg_score_vs_prediction_confidence.png")
    # Merge ESG Score with prediction confidence
    esg_confidence = pd.merge(aggregated_esg, esg_df[['Source PDF', 'Prediction Confidence']], on='Company', how='left')
    plot_scatter(
        data=esg_confidence,
        x='ESG Score',
        y='Prediction Confidence',
        title='ESG Score vs Prediction Confidence',
        xlabel='ESG Score (Total Sentences)',
        ylabel='Prediction Confidence',
        output_path=scatter_plot_path
    )
    
    # -------------------------------
    # Scatter Plot: Normalized ESG Score vs Prediction Confidence
    # -------------------------------
    
    print("\nPlotting Scatter Plot: Normalized ESG Score vs Prediction Confidence...")
    normalized_scatter_plot_path = os.path.join(eda_output_dir, "normalized_esg_score_vs_prediction_confidence.png")
    plot_scatter(
        data=esg_confidence,
        x='Normalized ESG Score',
        y='Prediction Confidence',
        title='Normalized ESG Score vs Prediction Confidence',
        xlabel='Normalized ESG Score (ESG Score / Total Words)',
        ylabel='Prediction Confidence',
        output_path=normalized_scatter_plot_path
    )
    
    # -------------------------------
    # Box Plot: ESG Score
    # -------------------------------
    
    print("\nPlotting Box Plot: ESG Score...")
    esg_score_boxplot_path = os.path.join(eda_output_dir, "esg_score_boxplot.png")
    plot_boxplot(
        data=aggregated_esg,
        column='ESG Score',
        title='Box Plot of ESG Scores',
        ylabel='ESG Score (Total Sentences)',
        output_path=esg_score_boxplot_path
    )
    
    # -------------------------------
    # Box Plot: Normalized ESG Score
    # -------------------------------
    
    print("\nPlotting Box Plot: Normalized ESG Score...")
    normalized_esg_score_boxplot_path = os.path.join(eda_output_dir, "normalized_esg_score_boxplot.png")
    plot_boxplot(
        data=aggregated_esg,
        column='Normalized ESG Score',
        title='Box Plot of Normalized ESG Scores',
        ylabel='Normalized ESG Score (ESG Score / Total Words)',
        output_path=normalized_esg_score_boxplot_path
    )
    
    print("\nEDA Visualization complete! All plots are saved in the 'eda_visualization_output/' directory.")

if __name__ == "__main__":
    main()
