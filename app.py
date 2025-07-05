from flask import Flask, render_template, request
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# Ensure NLTK downloads (run once)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load trained model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('best_model.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_features(text):
    char_count = len(text)
    word_count = len(text.split())
    capitals_count = sum(1 for c in text if c.isupper())
    special_char_count = sum(1 for c in text if not c.isalnum() and c != ' ')
    has_url = int('http' in text or 'www' in text)
    digit_count = sum(1 for c in text if c.isdigit())

    cleaned_text = clean_text(text)
    tfidf_features = vectorizer.transform([cleaned_text])

    # Create array of length-based features
    length_features = np.array([[char_count, word_count, capitals_count, special_char_count, has_url, digit_count]])

    # Combine TF-IDF and length features
    combined_features = hstack([tfidf_features, length_features])

    return combined_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    features = extract_features(message)
    prediction = model.predict(features)[0]

    result = "Spam" if prediction == 1 else "Ham"
    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)