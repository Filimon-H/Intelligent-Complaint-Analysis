import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_complaint_distribution(df, title="Distribution of Complaints by Product"):
    """
    Plots number of complaints per product.
    """
    product_counts = df['Product'].value_counts()

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x=product_counts.index, y=product_counts.values, palette="viridis")

    plt.title(title, fontsize=14)
    plt.xlabel("Product", fontsize=12)
    plt.ylabel("Number of Complaints", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def add_word_count_column(df, narrative_col="Consumer complaint narrative"):
    """
    Adds a 'word_count' column based on splitting by whitespace.
    """
    pd.options.display.float_format = '{:.2f}'.format
    df['narrative'] = df[narrative_col].fillna("")
    df['word_count'] = df['narrative'].apply(lambda x: len(x.split()))
    return df

def plot_word_count_distribution(df, max_x=500):
    """
    Plots histogram of word counts.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df['word_count'], bins=50, kde=True, color='teal')
    plt.title("Distribution of Word Counts in Complaint Narratives")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.xlim(0, max_x)
    plt.tight_layout()
    plt.show()

def describe_word_counts(df):
    """
    Prints summary statistics for the word_count column.
    """
    print("📊 Word count summary stats:")
    print(df['word_count'].describe())
