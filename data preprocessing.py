import re
import csv
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader

#journey starts

def pdf_to_text(pdf_path):
    """
    Extracts text from a single PDF file.
    """
    reader = PdfReader(pdf_path)
    all_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text.append(page_text)

    return "\n".join(all_text)



def clean_text(text):
    # Replace ? and ! with .
    text = text.replace('?', '.')
    text = text.replace('!', '.')
    text = text.replace(';', ',')


    # Keep only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)


    # Convert to lowercase
    text = text.lower()

    # Standardize certain phrases
    # Example: "carbon dioxide" -> "co2"
    text = text.replace("carbon dioxide", "co2")
    text = text.replcae("co 2", "co2")


    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


#here I aim to remove the stopwords so the text can be clean
def remove_stopwords(token_list, stop_words):
    return [word for word in token_list if word not in stop_words]



def lemmatize_tokens(token_list, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in token_list]





if __name__ == "__main__":

    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    pdf_folder = "/Users/chencfz/文件"  
    pdf_files = glob.glob(pdf_folder + "*.pdf")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    all_data = []

    for pdf_file in pdf_files:
        # Extract raw text for each PDF
        raw_text = pdf_to_text(pdf_file)

        # Create a DataFrame for this PDF's text
        # Assuming one big block of text per PDF file
        df = pd.DataFrame({"text": [raw_text]})

        # Clean text
        df['Cleaned Text'] = df['text'].apply(clean_text)

        # Tokenize
        df['Tokens'] = df['Cleaned Text'].apply(word_tokenize)

        # Remove stopwords
        df['Tokens Without Stopwords'] = df['Tokens'].apply(lambda tokens: remove_stopwords(tokens, stop_words))

        # Lemmatize
        df['Lemmatized Tokens'] = df['Tokens Without Stopwords'].apply(lambda tokens: lemmatize_tokens(tokens, lemmatizer))

        # Add the Source PDF column
        df['Source PDF'] = pdf_file

        all_data.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save to a single CSV file
    combined_df.to_csv("processed_data_all_combined.csv", index=False)

    print("Processing complete! Check 'processed_data_all_combined.csv' for the results.")
