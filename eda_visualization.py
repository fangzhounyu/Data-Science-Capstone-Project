import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Loads the dataset for EDA.
    Args:
        filepath (str): Path to the dataset.
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(filepath)
    return data

def generate_summary_statistics(data):
    """
    Generates summary statistics for the dataset.
    Args:
        data (pandas.DataFrame): Input dataset.
    """
    print("\nSummary Statistics:")
    print(data.describe())

def plot_distribution(data, column, output_filepath):
    """
    Plots the distribution of a specific column.
    Args:
        data (pandas.DataFrame): Input dataset.
        column (str): Column to visualize.
        output_filepath (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(output_filepath)
    plt.show()

def plot_correlation_heatmap(data, output_filepath):
    """
    Plots a heatmap of correlations between numeric variables.
    Args:
        data (pandas.DataFrame): Input dataset.
        output_filepath (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.savefig(output_filepath)
    plt.show()

def scatter_plot(data, x_column, y_column, output_filepath):
    """
    Creates a scatter plot to visualize the relationship between two variables.
    Args:
        data (pandas.DataFrame): Input dataset.
        x_column (str): Name of the x-axis variable.
        y_column (str): Name of the y-axis variable.
        output_filepath (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, alpha=0.7)
    plt.title(f"{x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(output_filepath)
    plt.show()

if __name__ == "__main__":
    # Filepath to the dataset
    dataset_filepath = "regression_data.csv"  # Replace with your dataset file path
    data = load_data(dataset_filepath)

    # Generate summary statistics
    print("Generating summary statistics...")
    generate_summary_statistics(data)

    # Visualize ESG score distribution
    print("Plotting ESG score distribution...")
    plot_distribution(data, "esg_score", "esg_score_distribution.png")

    # Visualize stock price change distribution
    print("Plotting stock price change distribution...")
    plot_distribution(data, "stock_price_change", "stock_price_change_distribution.png")

    # Plot correlation heatmap
    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(data, "correlation_heatmap.png")

    # Scatter plot of ESG score vs. stock price change
    print("Plotting ESG score vs. stock price change...")
    scatter_plot(data, "esg_score", "stock_price_change", "esg_vs_stock_price.png")
