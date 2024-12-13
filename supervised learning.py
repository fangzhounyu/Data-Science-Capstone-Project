import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(filepath):
    """
    Loads the labeled dataset for supervised learning.
    Args:
        filepath (str): Path to the labeled dataset.
    Returns:
        pandas.DataFrame: The labeled dataset.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_for_training(data, text_column, label_column):
    """
    Prepares the data for supervised learning by vectorizing text and splitting it into train/test sets.
    Args:
        data (pandas.DataFrame): Input dataset with text and labels.
        text_column (str): Column containing the text data.
        label_column (str): Column containing the labels.
    Returns:
        tuple: Train/test split of text features and labels.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Convert text to numerical features
    X = vectorizer.fit_transform(data[text_column]).toarray()
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(X_train, y_train):
    """
    Trains a supervised learning model (Random Forest) on the training data.
    Args:
        X_train (array): Training features.
        y_train (array): Training labels.
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (array): Test features.
        y_test (array): Test labels.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Filepath to the labeled dataset
    labeled_data_filepath = "labeled_data.csv"  # Replace with your actual labeled dataset

    # Load the labeled dataset
    print("Loading labeled dataset...")
    data = load_data(labeled_data_filepath)

    # Prepare data for training
    print("Preprocessing data for training...")
    text_column = "cleaned_text"  # Column with text data
    label_column = "esg_category"  # Column with labels
    X_train, X_test, y_train, y_test, vectorizer = preprocess_for_training(data, text_column, label_column)

    # Train the Random Forest model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the trained model and vectorizer
    joblib.dump(model, "random_forest_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved!")
