import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load ClimateBERT model and tokenizer from Hugging Face
MODEL_NAME = "climatebert/distilroberta-base-climate-detector"  # Replace with the desired ClimateBERT model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define a text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

def classify_text(text):
    """
    Classifies a piece of text into ESG categories using ClimateBERT.
    Args:
        text (str): The input text to classify.
    Returns:
        dict: A dictionary of categories and their confidence scores.
    """
    try:
        # Get predictions
        predictions = classifier(text)
        results = {pred['label']: pred['score'] for pred in predictions[0]}
        return results
    except Exception as e:
        print(f"Error processing text: {text[:30]}... | Error: {e}")
        return {}

def process_dataset(filepath, output_filepath):
    """
    Processes a dataset of ESG text using ClimateBERT.
    Args:
        filepath (str): Path to the input dataset (CSV).
        output_filepath (str): Path to save the processed dataset.
    """
    # Load the dataset
    data = pd.read_csv(filepath)

    # Ensure the required column exists
    if "cleaned_text" not in data.columns:
        raise ValueError("The dataset must include a 'cleaned_text' column with preprocessed text.")

    # Apply ClimateBERT classification to each text
    data['esg_metrics'] = data['cleaned_text'].apply(classify_text)

    # Extract top categories and their confidence scores
    data['top_category'] = data['esg_metrics'].apply(lambda x: max(x, key=x.get) if x else None)
    data['top_score'] = data['esg_metrics'].apply(lambda x: x[max(x, key=x.get)] if x else None)

    # Save the processed dataset
    data.to_csv(output_filepath, index=False)
    print(f"Processed dataset saved to {output_filepath}!")

if __name__ == "__main__":
    # Filepath to input and output datasets
    input_filepath = "train_data.csv"  # Preprocessed dataset with 'cleaned_text' column
    output_filepath = "climatebert_output.csv"

    # Process the dataset
    print("Processing dataset with ClimateBERT...")
    process_dataset(input_filepath, output_filepath)
    print("Done!")
