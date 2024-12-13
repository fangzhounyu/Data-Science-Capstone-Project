import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(filepath):
    """
    Loads the labeled dataset for hyperparameter tuning.
    Args:
        filepath (str): Path to the labeled dataset.
    Returns:
        pandas.DataFrame: The labeled dataset.
    """
    return pd.read_csv(filepath)

def preprocess_for_tuning(data, text_column, label_column):
    """
    Prepares data for hyperparameter tuning by vectorizing text and splitting into train/test sets.
    Args:
        data (pandas.DataFrame): Input dataset with text and labels.
        text_column (str): Column containing text data.
        label_column (str): Column containing labels.
    Returns:
        tuple: Train/test split of text features and labels.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Convert text to numerical features
    X = vectorizer.fit_transform(data[text_column]).toarray()
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, vectorizer

def perform_hyperparameter_tuning(X_train, y_train):
    """
    Tunes the hyperparameters of a Random Forest classifier using GridSearchCV.
    Args:
        X_train (array): Training features.
        y_train (array): Training labels.
    Returns:
        RandomForestClassifier: The best model with tuned hyperparameters.
    """
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Initialize Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the tuned model on the test data.
    Args:
        model (RandomForestClassifier): Tuned model.
        X_test (array): Test features.
        y_test (array): Test labels.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Filepath to the labeled dataset
    labeled_data_filepath = "labeled_data.csv"

    # Load the labeled dataset
    print("Loading labeled dataset...")
    data = load_data(labeled_data_filepath)

    # Prepare data for hyperparameter tuning
    print("Preprocessing data for hyperparameter tuning...")
    text_column = "cleaned_text"  # Column with text data
    label_column = "esg_category"  # Column with labels
    X_train, X_test, y_train, y_test, vectorizer = preprocess_for_tuning(data, text_column, label_column)

    # Perform hyperparameter tuning
    print("Performing hyperparameter tuning...")
    best_model = perform_hyperparameter_tuning(X_train, y_train)

    # Evaluate the tuned model
    print("Evaluating the tuned model...")
    evaluate_model(best_model, X_test, y_test)

    # Save the best model and vectorizer
    joblib.dump(best_model, "best_random_forest_model.pkl")
    joblib.dump(vectorizer, "best_tfidf_vectorizer.pkl")
    print("Best model and vectorizer saved!")
