import joblib
import numpy as np

def inspect_model():
    print("Loading model and vectorizer...")
    try:
        model = joblib.load('security_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except Exception as e:
        print(f"Error: {e}")
        return

    feature_names = vectorizer.get_feature_names_out()
    
    # Check if model is Random Forest or LinearSVC
    if hasattr(model, 'feature_importances_'):
        print("\nRandom Forest Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 20 most important words:")
        for i in range(20):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
            
    elif hasattr(model, 'coef_'):
        print("\nLinearSVC Coefficients:")
        for i, class_label in enumerate(model.classes_):
            print(f"\nClass: {class_label}")
            top10 = np.argsort(model.coef_[i])[-10:]
            for j in top10:
                print(f"  {feature_names[j]}")

if __name__ == "__main__":
    inspect_model()
