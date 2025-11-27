# Threat Level Integration - Using Original CSV

## ✅ What Changed

The `analysis.py` script now **calculates Threat Level on-the-fly** from the existing columns in your original `cyber_security.csv` file. No need for a separate enhanced CSV!

## How It Works

### Automatic Calculation

The script reads `Severity Score` and `Risk Level Prediction` from your existing CSV and calculates Threat Level using this logic:

```python
def calculate_threat_level(severity_score, risk_level):
    severity = int(severity_score)  # 1-5
    risk = int(risk_level)          # 1-5

    if severity >= 4 and risk >= 4:
        return 'Critical'  # Both high
    elif severity >= 4 or risk >= 4:
        return 'High'      # Either high
    elif severity <= 2 and risk <= 2:
        return 'Low'       # Both low
    else:
        return 'Medium'    # Everything else
```

### What the Model Does

1. **Loads** `cyber_security.csv` (original file)
2. **Calculates** Threat Level from existing columns
3. **Trains TWO models**:
   - Threat Category (DDoS, Malware, Phishing, Ransomware)
   - Threat Level (Critical, High, Medium, Low)
4. **Saves** three files:
   - `security_model.pkl`
   - `threat_level_model.pkl`
   - `tfidf_vectorizer.pkl`

## Usage

### Train Models

```bash
python3 analysis.py
```

### Use Flask API

```bash
python3 app.py
```

### Test Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ransomware encrypting files demanding bitcoin"}'
```

### Response

```json
{
  "threat_category": "Ransomware",
  "threat_level": "Critical",
  "recommendation": "Immediate action required! Escalate to security team."
}
```

## Files

### Modified

- ✅ `analysis.py` - Calculates threat level from existing data
- ✅ `app.py` - Returns both predictions

### No Longer Needed

- ❌ `cyber_security_enhanced.csv` - Not required
- ❌ `add_threat_level.py` - Not needed (can delete)

## Benefits

✅ **No CSV modification** - Uses original data
✅ **Dynamic calculation** - Threat level computed on-the-fly
✅ **Dual predictions** - Category + Level
✅ **Actionable recommendations** - Based on threat severity
✅ **Production ready** - Saved models for deployment
