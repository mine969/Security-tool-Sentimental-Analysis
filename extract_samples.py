import pandas as pd
import re

def get_correct_label(text):
    text = text.lower()
    if any(x in text for x in ['ransomware', 'encrypted', 'bitcoin', 'payment', 'locked', 'decrypt']):
        return 'Ransomware'
    elif any(x in text for x in ['phishing', 'email', 'link', 'click', 'password', 'login', 'account', 'credential']):
        return 'Phishing'
    elif any(x in text for x in ['ddos', 'traffic', 'overwhelmed', 'service', 'down', 'packet', 'flood', 'slow']):
        return 'DDoS'
    elif any(x in text for x in ['malware', 'trojan', 'virus', 'spyware', 'infected', 'backdoor', 'worm']):
        return 'Malware'
    elif any(x in text for x in ['sql', 'injection', 'database', 'query']):
        return 'SQL Injection'
    else:
        return 'Other'

def extract_samples():
    try:
        df = pd.read_csv('cyber_security.csv')
    except:
        print("Error reading CSV.")
        return

    # Apply label logic
    # Ensure column is string
    df['Cleaned Threat Description'] = df['Cleaned Threat Description'].astype(str)
    df['target_class'] = df['Cleaned Threat Description'].apply(get_correct_label)
    
    print("Label Counts:")
    print(df['target_class'].value_counts())
    
    categories = ['Malware', 'DDoS', 'Ransomware', 'Phishing']
    
    print("\n--- Extracted Samples ---")
    for cat in categories:
        sample = df[df['target_class'] == cat]['Cleaned Threat Description'].iloc[0]
        print(f"\nCategory: {cat}")
        print(f"Sample Text: {sample}")

if __name__ == "__main__":
    extract_samples()
