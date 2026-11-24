# EDA
from load_data import load_data, load_clean_data
from config import CLEAN_DATA_PATH

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def df_info(df: pd.DataFrame):
    print("Information About Data")
    pd.set_option('display.max_columns', None)

    print("\nShape:",df.shape)

    print("\nSample view of data:")
    print(df.sample(5))

    print("\nInfo about the data:")
    df.info()

    print("\nStatical Info about data:")
    print(df.describe())

    print("\nUnique Value counts of each columns:")
    for col in df.columns:
        print(col, df[col].value_counts())
        print('='*85)

    print("\nMissing Values:")
    missing_values = df.isnull().sum().sum()
    print("Total Missing Values",missing_values)
    if missing_values > 0:
        print(df.isnull().sum())

    print("\nDuplicate Values:")
    print(df.duplicated().sum())

def clean_data(df: pd.DataFrame):
    print('*'*85)

    print("Cleaning The Data...")

    print("\nDroping the colums...")
    drop_columns = ['customerID']
    df = df.drop(columns = drop_columns)

    print("Changing Data types...")
    df['TotalCharges'] = df['TotalCharges'].replace(' ', "0.0")
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['Churn'] = df['Churn'].map({"Yes": 1, "No": 0})

    df.to_csv(CLEAN_DATA_PATH, index = False)

    print('*'*85)

def plot_hist(df, column):
    plt.figure(figsize = (5, 3))
    sns.histplot(df[column], kde = True)
    plt.title(f"Distribution of {column}")

    col_mean   = df[column].mean()
    col_medium = df[column].median()

    plt.axvline(col_mean  , color = 'red',   label = 'Mean')
    plt.axvline(col_medium, color = 'green', label = 'Median')

    plt.legend()
    plt.show()

    # plt.savefig(f"{CHARTS_DIR}/Distribution of: {column}.png", dpi=300, bbox_inches="tight")

def plot_box(df, column):
    plt.figure(figsize = (5, 3))
    sns.boxplot(y = df[column])
    plt.title(f"Box-Plot of {column}")

    plt.legend()
    plt.show()

def eda(df):
    # Numerical
    plot_hist(df, 'tenure')
    plot_hist(df, 'MonthlyCharges')
    plot_hist(df, 'TotalCharges')

    plot_box(df, 'tenure')
    plot_box(df, 'MonthlyCharges')
    plot_box(df, 'TotalCharges')

    plt.figure(figsize = (8, 4))
    sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')
    plt.title('Correlation HeatMap')
    plt.show()


def main():
    df = load_data()

    df_info(df)

    clean_data(df)

    clean_df = load_clean_data()

    eda(clean_df)

if __name__ == "__main__":
    main()