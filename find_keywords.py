import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def find_keywords():
    print("Loading model...")
    try:
        model = joblib.load('security_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except:
        print("Error loading model.")
        return

    candidates = {
        "Malware": ["malware", "virus", "trojan", "spyware", "worm", "infected", "backdoor", "keylogger", "rootkit", "malicious code"],
        "DDoS": ["ddos", "denial", "service", "distributed", "flood", "traffic", "packet", "botnet", "overwhelmed", "slow", "down"]
    }

    print("\nTesting Candidates:")
    for category, words in candidates.items():
        print(f"\n--- Searching for {category} keywords ---")
        for word in words:
            clean = preprocess_text(word)
            vec = vectorizer.transform([clean])
            if vec.nnz == 0:
                print(f"'{word}' -> [Unknown/Empty Vector]")
                continue
                
            pred = model.predict(vec)[0]
            print(f"'{word}' -> Predicted: {pred}")
            
            if pred == category:
                print(f"  >>> FOUND MATCH: '{word}'")

if __name__ == "__main__":
    find_keywords()
