import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import logging



# -----------------------------------------
# Functions function function
# -----------------------------------------

def load_data(input_csv):
    """
    Load the ClimateBERT output CSV.
    """
    df = pd.read_csv(input_csv)
    return df



def preprocess_text(text):
    """
    Basic text preprocessing: remove non-alphabetic characters and lowercase the text.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()

    return text



def split_labeled_data(df, label_column, test_size=0.2, random_state=42):
    """
    Split the labeled data into training and validation sets.
    """
    labeled_df = df.dropna(subset=[label_column]).copy()

    X = labeled_df['Original Text'].apply(preprocess_text)
    y = labeled_df[label_column]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    

    return X_train, X_val, y_train, y_val



def train_tfidf_vectorizer(X_train):
    """
    Train a TF-IDF vectorizer on the training data.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    return vectorizer, X_train_tfidf



def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return rf

def train_random_forest_with_hyperparameter_tuning(X_train, y_train):
    """
    Train a Random Forest classifier with hyperparameter tuning using GridSearchCV.
    """
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    return best_rf

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and print average F1 scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Average F1 Score: {scores.mean():.4f}")

def evaluate_model(model, X_val, y_val, output_dir):
    """
    Evaluate the model and save evaluation metrics and confusion matrix.
    """
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print("Evaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    # Save classification report
    report = classification_report(y_val, y_pred, zero_division=0)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    

    print(f"Classification report saved to {os.path.join(output_dir, 'classification_report.txt')}")
    print(f"Confusion matrix saved to {cm_path}")



def save_artifacts(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and TF-IDF vectorizer.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


    print(f"Model saved to {model_path}")
    print(f"TF-IDF Vectorizer saved to {vectorizer_path}")



def apply_model_to_unlabeled(model, vectorizer, df, label_column, output_csv):
    """
    Apply the trained model to the unlabeled data and save predictions.
    """
    unlabeled_df = df[df[label_column].isna()].copy()
    X_unlabeled = unlabeled_df['Original Text'].apply(preprocess_text)
    X_unlabeled_tfidf = vectorizer.transform(X_unlabeled)
    
    predictions = model.predict(X_unlabeled_tfidf)
    prediction_probs = model.predict_proba(X_unlabeled_tfidf).max(axis=1)
    
    unlabeled_df['Predicted ESG Category'] = predictions
    unlabeled_df['Prediction Confidence'] = prediction_probs
    
    # Save predictions
    unlabeled_df.to_csv(output_csv, index=False)
    print(f"Predictions on unlabeled data saved to {output_csv}")


def plot_category_distribution(df, category_column, title, output_path):
    """
    Plot and save a bar chart of category distributions.
    """
    category_counts = df[category_column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title(title)
    plt.xlabel('ESG Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Category distribution plot saved to {output_path}")

def plot_feature_importance(model, vectorizer, output_dir, top_n=20):
    """
    Plot and save a bar chart of the top N feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    plt.close()
    print(f"Feature importances plot saved to {os.path.join(output_dir, 'feature_importances.png')}")



# -----------------------------------------
# Main Execution Block
# -----------------------------------------

def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths
    input_csv = "climatebert_output.csv"  # Input from the previous step
    output_dir = "supervised_learning_output"  # Directory to save outputs
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    predictions_csv = os.path.join(output_dir, "unlabeled_predictions.csv")
    
    # Label column name
    label_column = "Manual Label"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # Load Data
    # -------------------------------
    
    print("Loading data...")
    df = load_data(input_csv)
    print(f"Total records loaded: {len(df)}")
    
    # -------------------------------
    # Check for Manual Labels
    # -------------------------------
    
    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' column not found in the input CSV. add this column with manual labels for 20% of the data.")
    
    labeled_count = df[label_column].notna().sum()
    total_count = len(df)
    labeled_percentage = (labeled_count / total_count) * 100
    print(f"Labeled data: {labeled_count} records ({labeled_percentage:.2f}%)")
    
    if labeled_count < 0.2 * total_count:
        raise ValueError(f"Insufficient labeled data. Please ensure at least 20% of the data is manually labeled in the '{label_column}' column.")
    
    # -------------------------------
    # Split Labeled Data
    # -------------------------------
    
    print("Splitting labeled data into training and validation sets...")
    X_train, X_val, y_train, y_val = split_labeled_data(df, label_column)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # -------------------------------
    # Feature Extraction
    # -------------------------------
    
    print("Training TF-IDF vectorizer...")
    vectorizer, X_train_tfidf = train_tfidf_vectorizer(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    print("TF-IDF vectorization complete.")
    
    # -------------------------------
    # Model Training
    # -------------------------------
    
    print("Training Random Forest classifier...")
    rf_model = train_random_forest(X_train_tfidf, y_train)
    print("Model training complete.")
    
    # -------------------------------
    # Model Evaluation
    # -------------------------------
    
    print("Evaluating model...")
    evaluate_model(rf_model, X_val_tfidf, y_val, output_dir)
    
    # -------------------------------
    # Save Model and Vectorizer
    # -------------------------------
    
    print("Saving model and vectorizer...")
    save_artifacts(rf_model, vectorizer, model_path, vectorizer_path)
    
    # -------------------------------
    # Plot Feature Importance (Optional)
    # -------------------------------
    
    print("Plotting feature importances...")
    plot_feature_importance(rf_model, vectorizer, output_dir)
    
    # -------------------------------
    # Plot Category Distribution in Training and Validation Sets
    # -------------------------------
    
    print("Plotting category distribution in training set...")
    train_df = pd.DataFrame({'Original Text': X_train, 'ESG Category': y_train})
    plot_category_distribution(train_df, 'ESG Category', 
                               'Category Distribution in Training Set', 
                               os.path.join(output_dir, 'training_category_distribution.png'))
    
    print("Plotting category distribution in validation set...")
    val_df = pd.DataFrame({'Original Text': X_val, 'ESG Category': y_val})
    plot_category_distribution(val_df, 'ESG Category', 
                               'Category Distribution in Validation Set', 
                               os.path.join(output_dir, 'validation_category_distribution.png'))
    
    # -------------------------------
    # Apply Model to Unlabeled Data
    # -------------------------------
    
    print("Applying model to unlabeled data...")
    apply_model_to_unlabeled(rf_model, vectorizer, df, label_column, predictions_csv)
    
    # -------------------------------
    # Plot Category Distribution in Predictions
    # -------------------------------
    
    print("Plotting category distribution in predictions...")
    predictions_df = pd.read_csv(predictions_csv)
    plot_category_distribution(predictions_df, 'Predicted ESG Category', 
                               'Category Distribution in Unlabeled Predictions', 
                               os.path.join(output_dir, 'predictions_category_distribution.png'))
    
    print("Supervised learning pipeline complete!")

if __name__ == "__main__":
    main()
