from flask import Flask, request, jsonify
import openai
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv('secrets.env')

# Load the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the embeddings and processed lines
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('processed_lines.pkl', 'rb') as f:
    processed_lines = pickle.load(f)

# Function to retrieve the most relevant text based on a query
def retrieve_relevant_text(query, embeddings, texts, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return [texts[i] for i in top_indices]


@app.route('/')
def index():
    return "RAG Flask API is running."


# Route to handle incoming queries
@app.route('/query', methods=['GET','POST'])
def query():
    data = request.get_json()
    query = data['query']
    relevant_texts = retrieve_relevant_text(query, embeddings, processed_lines)
    
    response_texts = []
    for text in relevant_texts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text + "\n\n" + query}
            ],
            max_tokens=150
        )
        response_texts.append(response.choices[0].message['content'].strip())

    return jsonify({'responses': response_texts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
