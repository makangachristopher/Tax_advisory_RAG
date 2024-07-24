
# Import necessary libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define file path and load the JSON file
file_path = 'pdf_texts.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Assuming the JSON structure is {'texts': ["text1", "text2", ...]}
lines = data['Taxation_handbook_FY2022-23.pdf']

# Preprocess each line
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d', ' ', text)  # Remove digits
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords.words('english') and len(word) > 1]  # Remove stopwords and single characters
    return ' '.join(tokens)

processed_lines = [preprocess_text(line) for line in lines]

# Stem the words
porter_stemmer = PorterStemmer()
stemmed_lines = [' '.join(porter_stemmer.stem(word) for word in line.split()) for line in processed_lines]

# Print the processed lines to verify
print(stemmed_lines)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model based on your needs

# Convert the text data into embeddings
embeddings = model.encode(stemmed_lines)

# Save the embeddings and processed lines
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

with open('processed_lines.pkl', 'wb') as f:
    pickle.dump(stemmed_lines, f)
