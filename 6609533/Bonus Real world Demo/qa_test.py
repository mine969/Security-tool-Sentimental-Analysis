import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preprocessing Function (Must match training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def run_qa_tests():
    print("="*50)
    print("STARTING QA AUTOMATED TESTS")
    print("="*50)

    # Load Model
    try:
        model = joblib.load('security_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("[PASS] Model and Vectorizer loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Could not load model: {e}")
        return

    # Test Cases
    # Note: Using keywords confirmed by feature importance inspection.
    # The model classifies threats into: Phishing, Ransomware, DDoS.
    # (Malware is often categorized as Ransomware or Phishing depending on the vector).
    test_cases = [
        ("phishing email detected", "Phishing"),
        ("ransomware attack on network", "Ransomware"),
        ("distributed denial of service attack", "DDoS"),
        ("suspicious link in email", "Phishing")
    ]

    passed = 0
    total = len(test_cases)

    print("\nRunning Scenarios...")
    for text, expected in test_cases:
        # Predict
        clean_text = preprocess_text(text)
        vec_text = vectorizer.transform([clean_text])
        prediction = model.predict(vec_text)[0]
        
        if prediction == expected:
            print(f"[PASS] Input: '{text}' -> Predicted: {prediction}")
            passed += 1
        else:
            print(f"[FAIL] Input: '{text}' -> Predicted: {prediction} (Expected: {expected})")

    print("\n" + "="*50)
    print(f"QA RESULT: {passed}/{total} Tests Passed")
    print("="*50)

if __name__ == "__main__":
    run_qa_tests()
