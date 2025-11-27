4. **Run Automated QA Tests**:
   ```bash
   python qa_test.py
   ```
   _This runs a suite of automated tests to verify the model's performance on predefined scenarios._

## Files

- `analysis.py`: The main script for data processing and model training.
- `download_dataset.py`: Helper script to download data from Kaggle.
- `cyber_security.csv`: The dataset.
- `app.py`: The Web Dashboard backend.
- `qa_test.py`: Automated QA testing script.
- `security_model.pkl`: The trained model (saved for integration).
- `tfidf_vectorizer.pkl`: The feature extractor (saved for integration).

## Methodology Changes

The `analysis.py` script in this directory has been modified to:

1. **Rely Solely on Dataset Labels**: It uses the `Threat Category` column from `cyber_security.csv` directly for training.
2. **Remove Rule-Based Logic**: The hardcoded keyword matching (hybrid approach) has been removed to ensure the model learns purely from the provided data.
3. **Include All Categories**: No filtering of 'Other' categories (unless not present in dataset).
