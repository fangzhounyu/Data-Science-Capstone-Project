import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    """
    Cleans the input text by removing special characters, extra spaces, and stopwords.
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

# Load data
def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    Args:
        filepath (str): Path to the CSV file containing the raw data.
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(filepath)
    return data

# Preprocess data
def preprocess_data(data, text_column):
    """
    Preprocesses the text data for model input.
    Args:
        data (pandas.DataFrame): Data containing text to preprocess.
        text_column (str): Column name of the text data.
    Returns:
        pandas.DataFrame: Preprocessed data with cleaned text.
    """
    data['cleaned_text'] = data[text_column].apply(clean_text)
    return data

# Split dataset into train and test
def split_data(data, label_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    Args:
        data (pandas.DataFrame): Preprocessed dataset.
        label_column (str): Column name of the labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random split.
    Returns:
        tuple: Training and testing datasets.
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[label_column])
    return train, test

# Main function
if __name__ == "__main__":
    # Filepath to the dataset
    filepath = "raw_data.csv"  
    text_column = "esg_text"   # Column containing the raw text data
    label_column = "label"     # Column containing labels (if available)
    
    # Step 1: Load the data
    data = load_data(filepath)
    print("Data Loaded Successfully!")

    # Step 2: Preprocess the data
    preprocessed_data = preprocess_data(data, text_column)
    print("Data Preprocessing Completed!")
    
    # Step 3: Split the data into training and testing sets
    train_data, test_data = split_data(preprocessed_data, label_column)
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")
    
    # Save the preprocessed datasets
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    print("Preprocessed datasets saved!")
