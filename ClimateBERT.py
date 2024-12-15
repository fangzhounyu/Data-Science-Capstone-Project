import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import glob
import os

# -----------------------------------------
# again the code!
# -----------------------------------------


def load_climatebert_model(model_name):
    """
    Loads the ClimateBERT model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model



def classify_sentence(sentence, tokenizer, model, categories, device):
    """
    Classifies a single sentence into one of the ESG categories using ClimateBERT.
    Returns the predicted category and its confidence score.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1).squeeze()
        confidence, predicted_class = torch.max(probabilities, dim=0)

        predicted_category = categories[predicted_class.item()]
        confidence_score = confidence.item()

    return predicted_category, confidence_score

def split_into_sentences(text):
    """
    Splits text into sentences using NLTK's Punkt tokenizer.
    """
    return nltk.sent_tokenize(text)

# -----------------------------------------
# Main Execution Block
# -----------------------------------------

if __name__ == "__main__":
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Define the path to the preprocessed CSV
    input_csv = "processed_data_all_combined.csv"  # this is the output of previous step
    

    # Define the output CSV file
    output_csv = "climatebert_output.csv"
    

    # Define the ClimateBERT model name
    climatebert_model_name = "distilbert-base-uncased"  
    

    # Define ESG categories
    esg_categories = [
        "Emission Reduction Targets",

        "Renewable Energy Investments",
        "Sustainability Policies",
        "Water Usage Reduction",

        "Waste Recycling",

        "Green Bond Issuance",
        "Social Impact"
    ]
    
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------
    # Load Preprocessed Data
    # -------------------------------
    
    print("Loading preprocessed data...")
    df = pd.read_csv(input_csv)
    

    if 'Cleaned Text' not in df.columns or 'Source PDF' not in df.columns:
        raise ValueError("Input CSV must contain 'Cleaned Text' and 'Source PDF' columns.")
    
    # -------------------------------
    # Split Cleaned Text into Sentences
    # -------------------------------
    

    print("Splitting text into sentences...")


    # Create a new DataFrame to hold sentences and their source PDF
    sentences_data = []
    
    for index, row in df.iterrows():

        cleaned_text = row['Cleaned Text']
        source_pdf = row['Source PDF']
        
        # Split into sentences
        sentences = split_into_sentences(cleaned_text)
        
        for sentence in sentences:
            sentences_data.append({
                "Original Text": sentence,
                "Source PDF": os.path.basename(source_pdf)  # Extract filename without path
            })
    
    sentences_df = pd.DataFrame(sentences_data)

    print(f"Total sentences extracted: {len(sentences_df)}")
    

    # -------------------------------
    # Load ClimateBERT Model
    # -------------------------------
    
    print("Loading ClimateBERT model and tokenizer...")

    tokenizer, model = load_climatebert_model(climatebert_model_name)
    model.to(device)
    model.eval()
    
    # -------------------------------
    # Classify Each Sentence
    # -------------------------------
    
    print("Classifying sentences into ESG categories...")

    predicted_categories = []
    confidence_scores = []
    
    for idx, row in sentences_df.iterrows():
        sentence = row['Original Text']
        
        if isinstance(sentence, str) and len(sentence.strip()) > 0:
            try:
                category, score = classify_sentence(sentence, tokenizer, model, esg_categories, device)
            except Exception as e:
                print(f"Error classifying sentence at index {idx}: {e}")
                category, score = "Unknown", 0.0
        else:
            category, score = "Unknown", 0.0
        
        predicted_categories.append(category)
        confidence_scores.append(score)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} sentences...")
    
    
    # -------------------------------
    # Add Predictions to DataFrame
    # -------------------------------
    
    sentences_df['Predicted ESG Category'] = predicted_categories
    sentences_df['Confidence Score'] = confidence_scores
    
    # -------------------------------
    # Save to Output CSV
    # -------------------------------
    
    print(f"Saving classification results to '{output_csv}'...")
    sentences_df.to_csv(output_csv, index=False)
    
    print("ClimateBERT classification complete! Check 'climatebert_output.csv' for results.")
