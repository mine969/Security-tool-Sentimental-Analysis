from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask App
app = Flask(__name__)

# Load Model and Vectorizer
print("Loading model and vectorizer...")
try:
    model = joblib.load('security_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run analysis.py first to generate the model files.")
    exit()

# Preprocessing Function (Must match the one used in training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess
    clean_text = preprocess_text(text)
    # Vectorize
    vectorized_text = vectorizer.transform([clean_text])
    # Predict
    prediction = model.predict(vectorized_text)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
