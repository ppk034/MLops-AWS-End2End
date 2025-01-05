import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logger_config import logger

def save_combined_plot(plots, file_name):
    """
    Save multiple plots in a single image.
    :param plots: List of matplotlib figures
    :param file_name: Name of the output file
    """
    logger.info("Saving all plots in a single image...")
    fig, axes = plt.subplots(len(plots), 1, figsize=(12, 6 * len(plots)))
    if len(plots) == 1:
        axes = [axes]  # Ensure axes is a list for single plot
    for ax, plot_func in zip(axes, plots):
        plot_func(ax)
    plt.tight_layout()
    plt.savefig(f"../logs/{file_name}.png")
    plt.close()
    logger.info(f"Combined plots saved as {file_name}.png")

def eda(file_path):
    """
    Perform EDA on the given dataset.
    :param file_path: Path to the dataset file
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Convert 'date' to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            logger.info(f"'date' column converted to datetime.")

        # Log summary statistics
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Statistical Summary:\n{df.describe(include='all')}")

        # Univariate Analysis
        def univariate_plot(ax, column):
            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_title(f"{column} vs Distribution Values")

        # Bivariate Analysis (e.g., W/L vs PTS)
        def bivariate_plot(ax):
            sns.boxplot(x="w/l", y="pts", data=df, ax=ax)
            ax.set_title("W/L vs PTS")

        # Correlation Analysis
        def correlation_plot(ax):
            numeric_df = df.select_dtypes(include=["number"])
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")

        # Combine plots in a single image
        plots = [lambda ax, col=col: univariate_plot(ax, col) for col in df.select_dtypes(include=["number"]).columns]
        plots.append(bivariate_plot)
        plots.append(correlation_plot)
        save_combined_plot(plots, "eda_combined_plots")

    except Exception as e:
        logger.error(f"Error during EDA: {e}")

if __name__ == "__main__":
    file_path = "../data/project_data_cleaned.csv"
    eda(file_path)
