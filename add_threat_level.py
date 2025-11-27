#!/usr/bin/env python3
"""
Script to add Threat Level column and improve the cyber_security.csv file.
Threat Level is calculated based on Severity Score and Risk Level Prediction.
Uses only standard library (no external dependencies).
"""

import csv
from collections import Counter


def calculate_threat_level(severity_score, risk_level):
    """
    Calculate threat level based on severity score (1-5) and risk level (1-5).
    
    Threat Level Categories:
    - Critical: High severity (4-5) AND high risk (4-5)
    - High: High severity (4-5) OR high risk (4-5)
    - Medium: Medium severity (2-3) OR medium risk (2-3)
    - Low: Low severity (1) AND low risk (1-2)
    
    Args:
        severity_score: Integer from 1-5
        risk_level: Integer from 1-5
        
    Returns:
        String: 'Critical', 'High', 'Medium', or 'Low'
    """
    try:
        # Convert to integers to handle any type issues
        severity = int(severity_score)
        risk = int(risk_level)
        
        # Critical: Both high
        if severity >= 4 and risk >= 4:
            return 'Critical'
        
        # High: Either one is high
        elif severity >= 4 or risk >= 4:
            return 'High'
        
        # Low: Both are low
        elif severity <= 2 and risk <= 2:
            return 'Low'
        
        # Medium: Everything else
        else:
            return 'Medium'
    except (ValueError, TypeError):
        return 'Unknown'


def add_threat_level_column(input_file='cyber_security.csv', output_file='cyber_security_enhanced.csv'):
    """
    Read the CSV, add Threat Level column, and save to a new file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Reading {input_file}...")
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    
    print(f"Original columns: {len(headers)}")
    print(f"Total rows: {len(rows)}")
    
    # Find indices for Severity Score and Risk Level Prediction
    severity_col = 'Severity Score'
    risk_col = 'Risk Level Prediction'
    
    if severity_col not in headers or risk_col not in headers:
        print(f"Error: Required columns not found!")
        print(f"Available columns: {headers}")
        return
    
    # Add Threat Level to each row
    print("\nCalculating Threat Level...")
    threat_levels = []
    for row in rows:
        threat_level = calculate_threat_level(row[severity_col], row[risk_col])
        row['Threat Level'] = threat_level
        threat_levels.append(threat_level)
    
    # Create new headers with Threat Level after Risk Level Prediction
    new_headers = list(headers)
    risk_idx = new_headers.index(risk_col)
    new_headers.insert(risk_idx + 1, 'Threat Level')
    
    # Write to new file
    print(f"\nSaving enhanced data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Successfully saved! New columns: {len(new_headers)}")
    
    # Display statistics
    print("\n=== Threat Level Distribution ===")
    threat_counter = Counter(threat_levels)
    total = len(threat_levels)
    
    for level in ['Critical', 'High', 'Medium', 'Low', 'Unknown']:
        count = threat_counter.get(level, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{level:10s}: {count:4d} ({percentage:5.2f}%)")
    
    # Show some examples
    print("\n=== Sample Records (first 10) ===")
    print(f"{'Threat Category':<15} {'Severity':<10} {'Risk':<6} {'Threat Level':<12}")
    print("-" * 55)
    for i, row in enumerate(rows[:10]):
        print(f"{row.get('Threat Category', 'N/A'):<15} "
              f"{row.get(severity_col, 'N/A'):<10} "
              f"{row.get(risk_col, 'N/A'):<6} "
              f"{row.get('Threat Level', 'N/A'):<12}")
    
    # Threat Category breakdown
    print("\n=== Threat Level by Threat Category ===")
    category_threat = {}
    for row in rows:
        category = row.get('Threat Category', 'Unknown')
        threat = row.get('Threat Level', 'Unknown')
        
        if category not in category_threat:
            category_threat[category] = Counter()
        category_threat[category][threat] += 1
    
    for category in sorted(category_threat.keys()):
        print(f"\n{category}:")
        for level in ['Critical', 'High', 'Medium', 'Low']:
            count = category_threat[category].get(level, 0)
            print(f"  {level:10s}: {count:4d}")
    
    return rows


if __name__ == "__main__":
    # Run the enhancement
    enhanced_data = add_threat_level_column()
    
    print("\n" + "="*60)
    print("✓ Enhancement complete!")
    print("New file created: cyber_security_enhanced.csv")
    print("="*60)
