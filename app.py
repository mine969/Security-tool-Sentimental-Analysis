from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Initialize Flask App
app = Flask(__name__)

# Load Models and Vectorizer
print("="*60)
print("LOADING CYBERSECURITY THREAT ANALYSIS MODELS")
print("="*60)

try:
    model = joblib.load('security_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✓ Threat Category model loaded successfully")
except Exception as e:
    print(f"✗ Error loading threat category model: {e}")
    print("  Please run analysis.py first to generate the model files.")
    exit()

# Try to load Threat Level model (optional)
threat_level_model = None
try:
    if os.path.exists('threat_level_model.pkl'):
        threat_level_model = joblib.load('threat_level_model.pkl')
        print("✓ Threat Level model loaded successfully")
    else:
        print("⚠ Threat Level model not found (optional)")
        print("  Run analysis_enhanced.py to generate threat level predictions")
except Exception as e:
    print(f"⚠ Could not load threat level model: {e}")

print("="*60)
print(f"Models ready: Threat Category {'+ Threat Level' if threat_level_model else 'only'}")
print("="*60)

# Preprocessing Function (Must match the one used in training)
def preprocess_text(text):
    """Preprocess text for ML model"""
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
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict threat category and threat level for given text.
    
    Request JSON:
        {
            "text": "description of security threat"
        }
    
    Response JSON:
        {
            "threat_category": "Phishing",
            "threat_level": "High",  # if threat_level_model is available
            "confidence": "high"
        }
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Preprocess
        clean_text = preprocess_text(text)
        
        # Vectorize
        vectorized_text = vectorizer.transform([clean_text])
        
        # Predict Threat Category
        threat_category = model.predict(vectorized_text)[0]
        
        # Build response
        response = {
            'threat_category': threat_category,
            'text_analyzed': text[:100] + '...' if len(text) > 100 else text
        }
        
        # Predict Threat Level if model is available
        if threat_level_model:
            threat_level = threat_level_model.predict(vectorized_text)[0]
            response['threat_level'] = threat_level
            
            # Add recommendation based on threat level
            recommendations = {
                'Critical': 'Immediate action required! Escalate to security team.',
                'High': 'High priority - investigate and respond promptly.',
                'Medium': 'Monitor closely and apply standard security procedures.',
                'Low': 'Low priority - log for analysis and periodic review.'
            }
            response['recommendation'] = recommendations.get(threat_level, 'Review and assess.')
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'threat_category': True,
            'threat_level': threat_level_model is not None
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
